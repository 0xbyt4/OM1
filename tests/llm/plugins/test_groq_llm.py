"""Tests for Groq LLM plugin."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from llm import LLMConfig
from llm.output_model import Action, CortexOutputModel
from llm.plugins.groq_llm import GroqLLM


@pytest.fixture(autouse=True)
def mock_decorators():
    """Mock decorators used in GroqLLM."""

    def passthrough_decorator(*args, **kwargs):
        def decorator(func):
            return func

        if len(args) == 1 and callable(args[0]):
            return args[0]
        return decorator

    with (
        patch(
            "llm.plugins.groq_llm.AvatarLLMState.trigger_thinking",
            passthrough_decorator,
        ),
        patch(
            "llm.plugins.groq_llm.LLMHistoryManager.update_history",
            passthrough_decorator,
        ),
        patch("llm.plugins.groq_llm.LLMHistoryManager"),
    ):
        yield


class TestGroqLLMInit:
    """Tests for GroqLLM initialization."""

    def test_init_without_api_key_raises_error(self):
        """Test that initialization without api_key raises ValueError."""
        config = LLMConfig(api_key=None)

        with pytest.raises(ValueError, match="config file missing api_key"):
            with patch("llm.plugins.groq_llm.openai.AsyncOpenAI"):
                GroqLLM(config)

    def test_init_with_api_key_succeeds(self):
        """Test that initialization with api_key succeeds."""
        config = LLMConfig(api_key="test-api-key")

        with patch("llm.plugins.groq_llm.openai.AsyncOpenAI") as mock_client:
            llm = GroqLLM(config)

            mock_client.assert_called_once_with(
                base_url="https://api.openmind.org/api/core/groq",
                api_key="test-api-key",
            )
            assert llm._config.model == "llama-3.3-70b-versatile"

    def test_init_with_custom_model(self):
        """Test that custom model is preserved."""
        config = LLMConfig(api_key="test-api-key", model="llama-3.1-8b-instant")

        with patch("llm.plugins.groq_llm.openai.AsyncOpenAI"):
            llm = GroqLLM(config)

            assert llm._config.model == "llama-3.1-8b-instant"

    def test_init_with_custom_base_url(self):
        """Test that custom base_url is used."""
        custom_url = "https://api.groq.com/openai/v1"
        config = LLMConfig(api_key="test-api-key", base_url=custom_url)

        with patch("llm.plugins.groq_llm.openai.AsyncOpenAI") as mock_client:
            GroqLLM(config)

            mock_client.assert_called_once_with(
                base_url=custom_url,
                api_key="test-api-key",
            )


class TestGroqLLMAsk:
    """Tests for GroqLLM.ask method."""

    @pytest.fixture
    def groq_llm(self):
        """Create a GroqLLM instance for testing."""
        config = LLMConfig(api_key="test-api-key")

        with patch("llm.plugins.groq_llm.openai.AsyncOpenAI"):
            llm = GroqLLM(config)

        llm.io_provider = MagicMock()
        llm.io_provider.llm_start_time = None
        llm.io_provider.llm_end_time = None
        llm.function_schemas = []

        return llm

    @pytest.mark.asyncio
    async def test_ask_with_function_calls(self, groq_llm):
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

        groq_llm._client.chat.completions.create = AsyncMock(return_value=mock_response)

        with patch(
            "llm.plugins.groq_llm.convert_function_calls_to_actions"
        ) as mock_convert:
            mock_action = Action(type="speak", value="Hello world")
            mock_convert.return_value = [mock_action]

            result = await groq_llm.ask("Test prompt")

            assert result is not None
            assert isinstance(result, CortexOutputModel)
            assert len(result.actions) == 1
            mock_convert.assert_called_once()

    @pytest.mark.asyncio
    async def test_ask_without_function_calls(self, groq_llm):
        """Test ask method when model returns no function calls."""
        mock_message = MagicMock()
        mock_message.tool_calls = None

        mock_choice = MagicMock()
        mock_choice.message = mock_message

        mock_response = MagicMock()
        mock_response.choices = [mock_choice]

        groq_llm._client.chat.completions.create = AsyncMock(return_value=mock_response)

        result = await groq_llm.ask("Test prompt")

        assert result is None

    @pytest.mark.asyncio
    async def test_ask_formats_messages_correctly(self, groq_llm):
        """Test that ask method formats messages and calls API correctly."""
        mock_message = MagicMock()
        mock_message.tool_calls = None

        mock_choice = MagicMock()
        mock_choice.message = mock_message

        mock_response = MagicMock()
        mock_response.choices = [mock_choice]

        groq_llm._client.chat.completions.create = AsyncMock(return_value=mock_response)

        await groq_llm.ask("Test prompt")

        groq_llm._client.chat.completions.create.assert_called_once()
        call_args = groq_llm._client.chat.completions.create.call_args
        assert call_args.kwargs["model"] == "llama-3.3-70b-versatile"
        assert "messages" in call_args.kwargs

    @pytest.mark.asyncio
    async def test_ask_handles_api_error(self, groq_llm):
        """Test that API errors are handled gracefully."""
        groq_llm._client.chat.completions.create = AsyncMock(
            side_effect=Exception("API Error")
        )

        result = await groq_llm.ask("Test prompt")

        assert result is None

    @pytest.mark.asyncio
    async def test_ask_sets_io_provider_times(self, groq_llm):
        """Test that IO provider times are set correctly."""
        mock_message = MagicMock()
        mock_message.tool_calls = None

        mock_choice = MagicMock()
        mock_choice.message = mock_message

        mock_response = MagicMock()
        mock_response.choices = [mock_choice]

        groq_llm._client.chat.completions.create = AsyncMock(return_value=mock_response)

        await groq_llm.ask("Test prompt")

        assert groq_llm.io_provider.llm_start_time is not None
        assert groq_llm.io_provider.llm_end_time is not None
