"""
Fallback LLM implementation that provides automatic failover from cloud to local LLMs.

This module implements a fallback mechanism for LLM calls, automatically switching
from a primary (typically cloud-based) LLM to a fallback (typically local) LLM
when network issues or API errors occur.
"""

import asyncio
import logging
import time
import typing as T

import openai
from pydantic import Field

from llm import LLM, LLMConfig, load_llm
from providers.avatar_llm_state_provider import AvatarLLMState
from providers.llm_history_manager import LLMHistoryManager

R = T.TypeVar("R")


class NetworkStatus:
    """
    Tracks network connectivity status with exponential backoff for retries.

    Maintains state about whether the network is available and implements
    backoff logic to avoid hammering failing services.
    """

    def __init__(self, initial_backoff: float = 5.0, max_backoff: float = 300.0):
        """
        Initialize the NetworkStatus tracker.

        Parameters
        ----------
        initial_backoff : float
            Initial backoff time in seconds after a failure.
        max_backoff : float
            Maximum backoff time in seconds.
        """
        self._is_online = True
        self._last_failure_time: float = 0.0
        self._consecutive_failures = 0
        self._initial_backoff = initial_backoff
        self._max_backoff = max_backoff

    @property
    def is_online(self) -> bool:
        """
        Check if network is considered online.

        Returns True if either:
        - No failures have occurred
        - Enough time has passed since last failure to retry

        Returns
        -------
        bool
            True if network should be considered available.
        """
        if self._consecutive_failures == 0:
            return True

        backoff_time = min(
            self._initial_backoff * (2 ** (self._consecutive_failures - 1)),
            self._max_backoff,
        )
        elapsed = time.time() - self._last_failure_time
        return elapsed >= backoff_time

    def record_success(self) -> None:
        """Record a successful network operation, resetting failure state."""
        self._consecutive_failures = 0
        self._is_online = True
        logging.debug("Network status: Online (success recorded)")

    def record_failure(self) -> None:
        """Record a network failure, incrementing backoff."""
        self._consecutive_failures += 1
        self._last_failure_time = time.time()
        self._is_online = False
        backoff = min(
            self._initial_backoff * (2 ** (self._consecutive_failures - 1)),
            self._max_backoff,
        )
        logging.warning(
            f"Network status: Offline (failure #{self._consecutive_failures}, "
            f"next retry in {backoff:.1f}s)"
        )


class FallbackLLMConfig(LLMConfig):
    """
    Configuration for FallbackLLM.

    Parameters
    ----------
    primary_llm_type : str
        Class name of the primary LLM (default: "OpenAILLM").
    primary_llm_config : dict
        Configuration for the primary LLM.
    fallback_llm_type : str
        Class name of the fallback LLM (default: "OllamaLLM").
    fallback_llm_config : dict
        Configuration for the fallback LLM.
    primary_timeout : float
        Timeout in seconds for primary LLM calls.
    retry_primary_after : float
        Seconds to wait before retrying primary after failure.
    """

    primary_llm_type: str = Field(
        default="OpenAILLM", description="Class name of the primary LLM"
    )
    primary_llm_config: T.Dict[str, T.Any] = Field(
        default_factory=lambda: {"model": "gpt-4.1"},
        description="Configuration for the primary LLM",
    )
    fallback_llm_type: str = Field(
        default="OllamaLLM", description="Class name of the fallback LLM"
    )
    fallback_llm_config: T.Dict[str, T.Any] = Field(
        default_factory=lambda: {
            "model": "llama3.2",
            "base_url": "http://localhost:11434",
        },
        description="Configuration for the fallback LLM",
    )
    primary_timeout: float = Field(
        default=10.0, description="Timeout in seconds for primary LLM calls"
    )
    retry_primary_after: float = Field(
        default=30.0,
        description="Seconds to wait before retrying primary after failure",
    )


class FallbackLLM(LLM[R]):
    """
    LLM with automatic failover from primary to fallback.

    Attempts to use the primary LLM first. If it fails due to network issues,
    timeouts, or API errors, automatically falls back to a local LLM.

    Config example:
        "cortex_llm": {
            "type": "FallbackLLM",
            "config": {
                "primary_llm_type": "OpenAILLM",
                "primary_llm_config": {"model": "gpt-4.1"},
                "fallback_llm_type": "OllamaLLM",
                "fallback_llm_config": {"model": "llama3.2"}
            }
        }

    Parameters
    ----------
    config : FallbackLLMConfig
        Configuration settings for the Fallback LLM.
    available_actions : list[AgentAction], optional
        List of available actions for function calling.
    """

    NETWORK_ERRORS = (
        ConnectionError,
        TimeoutError,
        asyncio.TimeoutError,
        OSError,
    )

    def __init__(
        self,
        config: FallbackLLMConfig,
        available_actions: T.Optional[T.List] = None,
    ):
        """
        Initialize the FallbackLLM instance.

        Sets up primary and fallback LLMs based on configuration and initializes
        network status tracking.

        Parameters
        ----------
        config : FallbackLLMConfig
            Configuration settings for the Fallback LLM.
        available_actions : list[AgentAction], optional
            List of available actions for function calling.
        """
        super().__init__(config, available_actions)

        self._config: FallbackLLMConfig

        primary_type = self._config.primary_llm_type
        primary_cfg = self._config.primary_llm_config.copy()
        fallback_type = self._config.fallback_llm_type
        fallback_cfg = self._config.fallback_llm_config.copy()

        if self._config.api_key:
            primary_cfg["api_key"] = self._config.api_key

        self._primary_llm: LLM = load_llm(
            {"type": primary_type, "config": primary_cfg},
            available_actions=available_actions,
        )
        self._fallback_llm: LLM = load_llm(
            {"type": fallback_type, "config": fallback_cfg},
            available_actions=available_actions,
        )

        self._primary_llm._skip_state_management = True
        self._fallback_llm._skip_state_management = True

        self._network_status = NetworkStatus(
            initial_backoff=self._config.retry_primary_after
        )

        fallback_base_url = fallback_cfg.get("base_url", "http://localhost:11434")
        self._history_client = openai.AsyncClient(
            base_url=f"{fallback_base_url}/v1", api_key="local"
        )

        self.history_manager = LLMHistoryManager(self._config, self._history_client)

        logging.info(
            f"FallbackLLM initialized: primary={primary_type}, fallback={fallback_type}"
        )

    async def _call_with_timeout(
        self,
        llm: LLM,
        prompt: str,
        messages: T.List[T.Dict[str, T.Any]],
        timeout: float,
    ) -> T.Optional[R]:
        """
        Call an LLM with a timeout.

        Parameters
        ----------
        llm : LLM
            The LLM instance to call.
        prompt : str
            The prompt to send.
        messages : list of dict
            Conversation history.
        timeout : float
            Timeout in seconds.

        Returns
        -------
        R or None
            The LLM response, or None if failed/timed out.

        Raises
        ------
        asyncio.TimeoutError
            If the call times out.
        Exception
            If the LLM call fails.
        """
        return await asyncio.wait_for(llm.ask(prompt, messages), timeout=timeout)

    @AvatarLLMState.trigger_thinking()
    @LLMHistoryManager.update_history()
    async def ask(
        self, prompt: str, messages: T.List[T.Dict[str, T.Any]] = []
    ) -> T.Optional[R]:
        """
        Send prompt to LLM with automatic fallback on failure.

        First attempts to use the primary LLM. If it fails due to network issues
        or timeout, falls back to the local LLM. Implements exponential backoff
        for retrying the primary LLM after failures.

        Parameters
        ----------
        prompt : str
            The input prompt to send.
        messages : list of dict, optional
            Conversation history (default: []).

        Returns
        -------
        R or None
            Parsed response matching the output model, or None if both LLMs failed.
        """
        start_time = time.time()
        self.io_provider.llm_start_time = start_time
        self.io_provider.set_llm_prompt(prompt)

        result: T.Optional[R] = None
        used_fallback = False

        try:
            if self._network_status.is_online:
                try:
                    result = await self._call_with_timeout(
                        self._primary_llm,
                        prompt,
                        messages,
                        self._config.primary_timeout,
                    )
                    if result is not None:
                        self._network_status.record_success()
                        logging.debug(
                            f"Primary LLM responded in {time.time() - start_time:.2f}s"
                        )
                        return T.cast(R, result)
                except self.NETWORK_ERRORS as e:
                    logging.warning(f"Primary LLM network error: {e}")
                    self._network_status.record_failure()
                except Exception as e:
                    error_msg = str(e).lower()
                    if any(
                        keyword in error_msg
                        for keyword in [
                            "connection",
                            "timeout",
                            "network",
                            "unreachable",
                        ]
                    ):
                        logging.warning(f"Primary LLM connection error: {e}")
                        self._network_status.record_failure()
                    else:
                        logging.error(f"Primary LLM unexpected error: {e}")
                        raise
            else:
                logging.debug(
                    "Skipping primary LLM due to recent failures, using fallback"
                )

            logging.info("Using fallback LLM")
            used_fallback = True
            result = await self._fallback_llm.ask(prompt, messages)

            if result is not None:
                logging.debug(
                    f"Fallback LLM responded in {time.time() - start_time:.2f}s"
                )
                return T.cast(R, result)

            return None

        except Exception as e:
            logging.error(f"FallbackLLM error: {e}")
            return None

        finally:
            self.io_provider.llm_end_time = time.time()
            if used_fallback:
                logging.info(
                    f"FallbackLLM total time: {time.time() - start_time:.2f}s (used fallback)"
                )
