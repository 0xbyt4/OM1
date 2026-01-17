"""Tests for FallbackLLM plugin."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from llm.plugins.fallback_llm import FallbackLLM, FallbackLLMConfig, NetworkStatus


class TestNetworkStatus:
    """Tests for NetworkStatus class."""

    def test_initial_state_is_online(self):
        """Network should be considered online initially."""
        status = NetworkStatus()
        assert status.is_online is True

    def test_after_failure_goes_offline(self):
        """Network should go offline after recording failure."""
        status = NetworkStatus(initial_backoff=10.0)
        status.record_failure()
        assert status.is_online is False

    def test_success_resets_failures(self):
        """Recording success should reset failure count."""
        status = NetworkStatus(initial_backoff=10.0)
        status.record_failure()
        status.record_failure()
        assert status._consecutive_failures == 2

        status.record_success()
        assert status._consecutive_failures == 0
        assert status.is_online is True

    def test_exponential_backoff(self):
        """Backoff should increase exponentially with failures."""
        status = NetworkStatus(initial_backoff=5.0, max_backoff=300.0)

        status.record_failure()
        status.record_failure()
        status.record_failure()

        assert status._consecutive_failures == 3
        # After 3 failures: 5.0 * (2 ** (3 - 1)) = 20.0
        expected_backoff = 20.0
        backoff = min(5.0 * (2 ** (3 - 1)), 300.0)
        assert backoff == expected_backoff

    def test_max_backoff_cap(self):
        """Backoff should not exceed max_backoff."""
        status = NetworkStatus(initial_backoff=100.0, max_backoff=300.0)

        for _ in range(10):
            status.record_failure()

        backoff = min(100.0 * (2 ** (10 - 1)), 300.0)
        assert backoff == 300.0


class TestFallbackLLMConfig:
    """Tests for FallbackLLMConfig."""

    def test_default_values(self):
        """Config should have sensible defaults."""
        config = FallbackLLMConfig()

        assert config.primary_llm_type == "OpenAILLM"
        assert config.fallback_llm_type == "OllamaLLM"
        assert config.primary_timeout == 10.0
        assert config.retry_primary_after == 30.0

    def test_custom_values(self):
        """Config should accept custom values."""
        config = FallbackLLMConfig(
            primary_llm_type="GeminiLLM",
            primary_llm_config={"model": "gemini-2.0-flash"},
            fallback_llm_type="QwenLLM",
            fallback_llm_config={"model": "qwen-2.5"},
            primary_timeout=5.0,
            retry_primary_after=60.0,
        )

        assert config.primary_llm_type == "GeminiLLM"
        assert config.fallback_llm_type == "QwenLLM"
        assert config.primary_timeout == 5.0
        assert config.retry_primary_after == 60.0


class TestFallbackLLM:
    """Tests for FallbackLLM class."""

    @pytest.fixture
    def mock_llm_classes(self):
        """Mock the load_llm function."""
        with patch("llm.plugins.fallback_llm.load_llm") as mock_load_llm:
            primary_instance = MagicMock()
            fallback_instance = MagicMock()

            def load_llm_side_effect(config, available_actions=None):
                if config["type"] == "OpenAILLM":
                    primary_instance._config = MagicMock()
                    primary_instance._config.api_key = config["config"].get("api_key")
                    return primary_instance
                elif config["type"] == "OllamaLLM":
                    return fallback_instance
                return MagicMock()

            mock_load_llm.side_effect = load_llm_side_effect

            yield {
                "load_llm": mock_load_llm,
                "primary_instance": primary_instance,
                "fallback_instance": fallback_instance,
            }

    @pytest.fixture
    def mock_openai_client(self):
        """Mock the OpenAI AsyncClient."""
        with patch("llm.plugins.fallback_llm.openai.AsyncClient") as mock_client:
            yield mock_client

    @pytest.fixture
    def mock_history_manager(self):
        """Mock the LLMHistoryManager."""
        with patch("llm.plugins.fallback_llm.LLMHistoryManager") as mock_hm:
            yield mock_hm

    def test_initialization(
        self, mock_llm_classes, mock_openai_client, mock_history_manager
    ):
        """FallbackLLM should initialize both primary and fallback LLMs."""
        config = FallbackLLMConfig()

        _llm = FallbackLLM(config=config, available_actions=[])

        assert mock_llm_classes["load_llm"].call_count == 2
        assert mock_history_manager.called
        assert _llm is not None

    @pytest.mark.asyncio
    async def test_uses_primary_when_online(
        self, mock_llm_classes, mock_openai_client, mock_history_manager
    ):
        """Should use primary LLM when network is available."""
        config = FallbackLLMConfig()
        llm = FallbackLLM(config=config, available_actions=[])

        expected_response = MagicMock()
        expected_response.actions = []

        llm._primary_llm.ask = AsyncMock(return_value=expected_response)
        llm._fallback_llm.ask = AsyncMock(return_value=None)
        llm._skip_state_management = True

        result = await llm.ask("test prompt")

        assert result == expected_response
        llm._primary_llm.ask.assert_called_once()
        llm._fallback_llm.ask.assert_not_called()

    @pytest.mark.asyncio
    async def test_falls_back_on_network_error(
        self, mock_llm_classes, mock_openai_client, mock_history_manager
    ):
        """Should fall back to local LLM on network errors."""
        config = FallbackLLMConfig()
        llm = FallbackLLM(config=config, available_actions=[])

        fallback_response = MagicMock()
        fallback_response.actions = []

        llm._primary_llm.ask = AsyncMock(side_effect=ConnectionError("No network"))
        llm._fallback_llm.ask = AsyncMock(return_value=fallback_response)
        llm._skip_state_management = True

        result = await llm.ask("test prompt")

        assert result == fallback_response
        llm._fallback_llm.ask.assert_called_once()

    @pytest.mark.asyncio
    async def test_falls_back_on_timeout(
        self, mock_llm_classes, mock_openai_client, mock_history_manager
    ):
        """Should fall back to local LLM on timeout."""
        config = FallbackLLMConfig(primary_timeout=0.1)
        llm = FallbackLLM(config=config, available_actions=[])

        fallback_response = MagicMock()
        fallback_response.actions = []

        async def slow_primary(*args, **kwargs):
            await asyncio.sleep(1.0)
            return MagicMock()

        llm._primary_llm.ask = slow_primary
        llm._fallback_llm.ask = AsyncMock(return_value=fallback_response)
        llm._skip_state_management = True

        result = await llm.ask("test prompt")

        assert result == fallback_response

    @pytest.mark.asyncio
    async def test_skips_primary_after_failures(
        self, mock_llm_classes, mock_openai_client, mock_history_manager
    ):
        """Should skip primary LLM after recent failures."""
        config = FallbackLLMConfig(retry_primary_after=60.0)
        llm = FallbackLLM(config=config, available_actions=[])

        fallback_response = MagicMock()
        fallback_response.actions = []

        llm._primary_llm.ask = AsyncMock(side_effect=ConnectionError("No network"))
        llm._fallback_llm.ask = AsyncMock(return_value=fallback_response)
        llm._skip_state_management = True

        await llm.ask("prompt 1")

        llm._primary_llm.ask.reset_mock()
        llm._fallback_llm.ask.reset_mock()

        await llm.ask("prompt 2")

        llm._primary_llm.ask.assert_not_called()
        llm._fallback_llm.ask.assert_called_once()

    @pytest.mark.asyncio
    async def test_retries_primary_after_success(
        self, mock_llm_classes, mock_openai_client, mock_history_manager
    ):
        """Should retry primary LLM after successful call."""
        config = FallbackLLMConfig()
        llm = FallbackLLM(config=config, available_actions=[])

        primary_response = MagicMock()
        primary_response.actions = []

        llm._primary_llm.ask = AsyncMock(return_value=primary_response)
        llm._fallback_llm.ask = AsyncMock(return_value=None)
        llm._skip_state_management = True

        llm._network_status.record_failure()

        llm._network_status._last_failure_time = 0

        result = await llm.ask("test prompt")

        assert result == primary_response
        llm._primary_llm.ask.assert_called_once()

    @pytest.mark.asyncio
    async def test_falls_back_on_connection_keyword_error(
        self, mock_llm_classes, mock_openai_client, mock_history_manager
    ):
        """Should fall back when error message contains connection keywords."""
        config = FallbackLLMConfig()
        llm = FallbackLLM(config=config, available_actions=[])

        fallback_response = MagicMock()
        fallback_response.actions = []

        llm._primary_llm.ask = AsyncMock(
            side_effect=Exception("Connection refused to server")
        )
        llm._fallback_llm.ask = AsyncMock(return_value=fallback_response)
        llm._skip_state_management = True

        result = await llm.ask("test prompt")

        assert result == fallback_response
        llm._fallback_llm.ask.assert_called_once()

    @pytest.mark.asyncio
    async def test_unexpected_error_returns_none(
        self, mock_llm_classes, mock_openai_client, mock_history_manager
    ):
        """Should return None on unexpected errors to prevent system crash."""
        config = FallbackLLMConfig()
        llm = FallbackLLM(config=config, available_actions=[])

        llm._primary_llm.ask = AsyncMock(side_effect=ValueError("Invalid input"))
        llm._fallback_llm.ask = AsyncMock(return_value=None)
        llm._skip_state_management = True

        result = await llm.ask("test prompt")
        assert result is None

    @pytest.mark.asyncio
    async def test_returns_none_when_both_fail(
        self, mock_llm_classes, mock_openai_client, mock_history_manager
    ):
        """Should return None when both LLMs fail."""
        config = FallbackLLMConfig()
        llm = FallbackLLM(config=config, available_actions=[])

        llm._primary_llm.ask = AsyncMock(side_effect=ConnectionError("No network"))
        llm._fallback_llm.ask = AsyncMock(return_value=None)
        llm._skip_state_management = True

        result = await llm.ask("test prompt")

        assert result is None

    def test_api_key_passed_to_primary(
        self, mock_llm_classes, mock_openai_client, mock_history_manager
    ):
        """Should pass api_key to primary LLM config."""
        config = FallbackLLMConfig(api_key="test-api-key")
        _llm = FallbackLLM(config=config, available_actions=[])

        calls = mock_llm_classes["load_llm"].call_args_list
        primary_call = calls[0]
        assert primary_call[0][0]["config"]["api_key"] == "test-api-key"
        assert _llm is not None
