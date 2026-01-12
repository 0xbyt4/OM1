"""Tests for Claude LLM plugin."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from llm import LLMConfig
from llm.output_model import Action, CortexOutputModel
from llm.plugins.claude_llm import ClaudeLLM


@pytest.fixture(autouse=True)
def mock_avatar_components():
    """Mock all avatar and IO components to prevent Zenoh session creation."""

    def mock_decorator(func=None):
        def decorator(f):
            return f

        if func is not None:
            return decorator(func)
        return decorator

    with (
        patch(
            "llm.plugins.claude_llm.AvatarLLMState.trigger_thinking",
            mock_decorator,
        ),
        patch(
            "llm.plugins.claude_llm.LLMHistoryManager.update_history",
            mock_decorator,
        ),
        patch("llm.plugins.claude_llm.LLMHistoryManager"),
        patch("llm.plugins.claude_llm.AvatarLLMState") as mock_avatar_state,
        patch("providers.avatar_provider.AvatarProvider") as mock_avatar_provider,
        patch(
            "providers.avatar_llm_state_provider.AvatarProvider"
        ) as mock_avatar_llm_state_provider,
    ):
        mock_avatar_state._instance = None
        mock_avatar_state._lock = None

        mock_provider_instance = MagicMock()
        mock_provider_instance.running = False
        mock_provider_instance.session = None
        mock_provider_instance.stop = MagicMock()
        mock_avatar_provider.return_value = mock_provider_instance
        mock_avatar_llm_state_provider.return_value = mock_provider_instance

        yield


class TestClaudeLLMInit:
    """Tests for ClaudeLLM initialization."""

    def test_init_without_api_key_raises_error(self):
        """Test that initialization without api_key raises ValueError."""
        config = LLMConfig(api_key=None)

        with pytest.raises(ValueError, match="config file missing api_key"):
            with patch("llm.plugins.claude_llm.openai.AsyncOpenAI"):
                ClaudeLLM(config)

    def test_init_with_api_key_succeeds(self):
        """Test that initialization with api_key succeeds."""
        config = LLMConfig(api_key="test-api-key")

        with patch("llm.plugins.claude_llm.openai.AsyncOpenAI") as mock_client:
            llm = ClaudeLLM(config)

            mock_client.assert_called_once_with(
                base_url="https://api.openmind.org/api/core/claude",
                api_key="test-api-key",
            )
            assert llm._config.model == "claude-sonnet-4-5-20250929"

    def test_init_with_custom_model(self):
        """Test that custom model is preserved."""
        config = LLMConfig(api_key="test-api-key", model="claude-opus-4-5-20251101")

        with patch("llm.plugins.claude_llm.openai.AsyncOpenAI"):
            llm = ClaudeLLM(config)

            assert llm._config.model == "claude-opus-4-5-20251101"

    def test_init_with_custom_base_url(self):
        """Test that custom base_url is used."""
        custom_url = "https://api.anthropic.com/v1"
        config = LLMConfig(api_key="test-api-key", base_url=custom_url)

        with patch("llm.plugins.claude_llm.openai.AsyncOpenAI") as mock_client:
            ClaudeLLM(config)

            mock_client.assert_called_once_with(
                base_url=custom_url,
                api_key="test-api-key",
            )


class TestClaudeLLMAsk:
    """Tests for ClaudeLLM.ask method."""

    @pytest.fixture
    def claude_llm(self):
        """Create a ClaudeLLM instance for testing."""
        config = LLMConfig(api_key="test-api-key")

        with patch("llm.plugins.claude_llm.openai.AsyncOpenAI"):
            llm = ClaudeLLM(config)

        llm.io_provider = MagicMock()
        llm.io_provider.llm_start_time = None
        llm.io_provider.llm_end_time = None
        llm.function_schemas = []

        return llm

    @pytest.mark.asyncio
    async def test_ask_with_function_calls(self, claude_llm):
        """Test ask method when model returns function calls."""
        mock_tool_call = MagicMock()
        mock_tool_call.function.name = "speak"
        mock_tool_call.function.arguments = '{"action": "Hello world"}'

        mock_message = MagicMock()
        mock_message.tool_calls = [mock_tool_call]

        mock_choice = MagicMock()
        mock_choice.message = mock_message

        mock_response = MagicMock()
        mock_response.choices = [mock_choice]

        claude_llm._client.chat.completions.create = AsyncMock(
            return_value=mock_response
        )

        with patch(
            "llm.plugins.claude_llm.convert_function_calls_to_actions"
        ) as mock_convert:
            mock_action = Action(type="speak", value="Hello world")
            mock_convert.return_value = [mock_action]

            result = await claude_llm.ask("Test prompt")

            assert result is not None
            assert isinstance(result, CortexOutputModel)
            assert len(result.actions) == 1
            mock_convert.assert_called_once()

    @pytest.mark.asyncio
    async def test_ask_without_function_calls(self, claude_llm):
        """Test ask method when model returns no function calls."""
        mock_message = MagicMock()
        mock_message.tool_calls = None

        mock_choice = MagicMock()
        mock_choice.message = mock_message

        mock_response = MagicMock()
        mock_response.choices = [mock_choice]

        claude_llm._client.chat.completions.create = AsyncMock(
            return_value=mock_response
        )

        result = await claude_llm.ask("Test prompt")

        assert result is None

    @pytest.mark.asyncio
    async def test_ask_formats_messages_correctly(self, claude_llm):
        """Test that ask method formats messages and calls API correctly."""
        mock_message = MagicMock()
        mock_message.tool_calls = None

        mock_choice = MagicMock()
        mock_choice.message = mock_message

        mock_response = MagicMock()
        mock_response.choices = [mock_choice]

        claude_llm._client.chat.completions.create = AsyncMock(
            return_value=mock_response
        )

        await claude_llm.ask("Test prompt")

        claude_llm._client.chat.completions.create.assert_called_once()
        call_args = claude_llm._client.chat.completions.create.call_args
        assert call_args.kwargs["model"] == "claude-sonnet-4-5-20250929"
        assert "messages" in call_args.kwargs

    @pytest.mark.asyncio
    async def test_ask_handles_api_error(self, claude_llm):
        """Test that API errors are handled gracefully."""
        claude_llm._client.chat.completions.create = AsyncMock(
            side_effect=Exception("API Error")
        )

        result = await claude_llm.ask("Test prompt")

        assert result is None

    @pytest.mark.asyncio
    async def test_ask_sets_io_provider_times(self, claude_llm):
        """Test that IO provider times are set correctly."""
        mock_message = MagicMock()
        mock_message.tool_calls = None

        mock_choice = MagicMock()
        mock_choice.message = mock_message

        mock_response = MagicMock()
        mock_response.choices = [mock_choice]

        claude_llm._client.chat.completions.create = AsyncMock(
            return_value=mock_response
        )

        await claude_llm.ask("Test prompt")

        assert claude_llm.io_provider.llm_start_time is not None
        assert claude_llm.io_provider.llm_end_time is not None
