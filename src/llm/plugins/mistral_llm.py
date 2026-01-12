"""Mistral LLM plugin for Mistral AI models with function calling support."""

import logging
import time
import typing as T

import openai
from pydantic import BaseModel

from llm import LLM, LLMConfig
from llm.function_schemas import convert_function_calls_to_actions
from llm.output_model import CortexOutputModel
from providers.avatar_llm_state_provider import AvatarLLMState
from providers.llm_history_manager import LLMHistoryManager

R = T.TypeVar("R", bound=BaseModel)


class MistralLLM(LLM[R]):
    """
    Mistral AI Language Model implementation using OpenAI-compatible API.

    Mistral AI provides efficient, high-performance models with excellent
    price-to-performance ratio. Supported models include:
    - mistral-large-latest: Frontier model with 675B parameters (recommended)
    - mistral-medium-latest: GPT-4 class performance, cost-effective
    - mistral-small-latest: Fast and efficient for simple tasks
    - magistral-medium-latest: Enhanced reasoning capabilities

    This class implements the LLM interface for Mistral models, handling
    configuration, authentication, and async API communication.

    Parameters
    ----------
    config : LLMConfig
        Configuration object containing API settings.
    available_actions : list[AgentAction], optional
        List of available actions for function call generation. If provided,
        the LLM will use function calls instead of structured JSON output.
    """

    def __init__(
        self,
        config: LLMConfig,
        available_actions: T.Optional[T.List] = None,
    ):
        """
        Initialize the Mistral LLM instance.

        Parameters
        ----------
        config : LLMConfig
            Configuration settings for the LLM.
        available_actions : list[AgentAction], optional
            List of available actions for function calling.
        """
        super().__init__(config, available_actions)

        if not config.api_key:
            raise ValueError("config file missing api_key")
        if not config.model:
            self._config.model = "mistral-large-latest"

        self._client = openai.AsyncOpenAI(
            base_url=config.base_url or "https://api.openmind.org/api/core/mistral",
            api_key=config.api_key,
        )

        self.history_manager = LLMHistoryManager(self._config, self._client)

    @AvatarLLMState.trigger_thinking()
    @LLMHistoryManager.update_history()
    async def ask(
        self, prompt: str, messages: T.List[T.Dict[str, str]] = []
    ) -> T.Optional[R]:
        """
        Send a prompt to the Mistral API and get a structured response.

        Parameters
        ----------
        prompt : str
            The input prompt to send to the model.
        messages : List[Dict[str, str]]
            List of message dictionaries containing conversation history.

        Returns
        -------
        R or None
            Parsed response matching the output_model structure, or None if
            parsing fails or an error occurs.
        """
        try:
            logging.debug(f"Mistral LLM input: {prompt}")
            logging.debug(f"Mistral LLM messages: {messages}")

            self.io_provider.llm_start_time = time.time()
            self.io_provider.set_llm_prompt(prompt)

            formatted_messages = [
                {"role": msg.get("role", "user"), "content": msg.get("content", "")}
                for msg in messages
            ]
            formatted_messages.append({"role": "user", "content": prompt})

            response = await self._client.chat.completions.create(
                model=self._config.model or "mistral-large-latest",
                messages=T.cast(T.Any, formatted_messages),
                tools=T.cast(T.Any, self.function_schemas),
                tool_choice="auto",
                timeout=self._config.timeout,
            )

            message = response.choices[0].message
            self.io_provider.llm_end_time = time.time()

            if message.tool_calls:
                logging.info(f"Received {len(message.tool_calls)} function calls")
                logging.info(f"Function calls: {message.tool_calls}")

                function_call_data = [
                    {
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        }
                    }
                    for tc in message.tool_calls
                ]

                actions = convert_function_calls_to_actions(function_call_data)

                result = CortexOutputModel(actions=actions)
                logging.info(f"Mistral LLM function call output: {result}")
                return T.cast(R, result)

            return None
        except Exception as e:
            logging.error(f"Mistral API error: {e}")
            return None
