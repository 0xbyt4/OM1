"""Tests for GLM-4 LLM plugin."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from llm import LLMConfig
from llm.output_model import Action, CortexOutputModel
from llm.plugins.glm4_llm import (
    DEFAULT_GLM4_BASE_URL,
    DEFAULT_GLM4_MODEL,
    GLM4LLM,
)


@pytest.fixture(autouse=True)
def mock_decorators():
    """Mock decorators used in GLM4LLM."""

    def passthrough_decorator(*args, **kwargs):
        def decorator(func):
            return func

        if len(args) == 1 and callable(args[0]):
            return args[0]
        return decorator

    with (
        patch(
            "llm.plugins.glm4_llm.AvatarLLMState.trigger_thinking",
            passthrough_decorator,
        ),
        patch(
            "llm.plugins.glm4_llm.LLMHistoryManager.update_history",
            passthrough_decorator,
        ),
        patch("llm.plugins.glm4_llm.LLMHistoryManager"),
    ):
        yield


class TestGLM4LLMConfig:
    """Tests for GLM4LLM configuration."""

    def test_default_model(self):
        """Test that default model is glm-4.7."""
        assert DEFAULT_GLM4_MODEL == "glm-4.7"

    def test_default_base_url(self):
        """Test that default base URL is OpenMind proxy."""
        assert DEFAULT_GLM4_BASE_URL == "https://api.openmind.org/api/core/glm4"


class TestGLM4LLMInit:
    """Tests for GLM4LLM initialization."""

    def test_init_without_api_key_raises_error(self):
        """Test that initialization without api_key raises ValueError."""
        config = LLMConfig(api_key=None)

        with pytest.raises(ValueError, match="config file missing api_key"):
            with patch("llm.plugins.glm4_llm.openai.AsyncOpenAI"):
                GLM4LLM(config)

    def test_init_with_api_key_succeeds(self):
        """Test that initialization with api_key succeeds."""
        config = LLMConfig(api_key="test-api-key")

        with patch("llm.plugins.glm4_llm.openai.AsyncOpenAI") as mock_client:
            llm = GLM4LLM(config)

            mock_client.assert_called_once_with(
                base_url=DEFAULT_GLM4_BASE_URL,
                api_key="test-api-key",
            )
            assert llm._config.model == DEFAULT_GLM4_MODEL

    def test_init_with_custom_model(self):
        """Test that custom model is preserved."""
        config = LLMConfig(api_key="test-api-key", model="glm-4-flash")

        with patch("llm.plugins.glm4_llm.openai.AsyncOpenAI"):
            llm = GLM4LLM(config)

            assert llm._config.model == "glm-4-flash"

    def test_init_with_custom_base_url(self):
        """Test that custom base_url is used."""
        custom_url = "https://custom.api.example.com/v1/"
        config = LLMConfig(api_key="test-api-key", base_url=custom_url)

        with patch("llm.plugins.glm4_llm.openai.AsyncOpenAI") as mock_client:
            GLM4LLM(config)

            mock_client.assert_called_once_with(
                base_url=custom_url,
                api_key="test-api-key",
            )


class TestGLM4LLMAsk:
    """Tests for GLM4LLM.ask method."""

    @pytest.fixture
    def glm4_llm(self):
        """Create a GLM4LLM instance for testing."""
        config = LLMConfig(api_key="test-api-key")

        with patch("llm.plugins.glm4_llm.openai.AsyncOpenAI"):
            llm = GLM4LLM(config)

        llm.io_provider = MagicMock()
        llm.io_provider.llm_start_time = None
        llm.io_provider.llm_end_time = None
        llm.function_schemas = []

        return llm

    @pytest.mark.asyncio
    async def test_ask_with_function_calls(self, glm4_llm):
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

        glm4_llm._client.chat.completions.create = AsyncMock(return_value=mock_response)

        with patch(
            "llm.plugins.glm4_llm.convert_function_calls_to_actions"
        ) as mock_convert:
            mock_action = Action(type="speak", value="Hello world")
            mock_convert.return_value = [mock_action]

            result = await glm4_llm.ask("Test prompt")

            assert result is not None
            assert isinstance(result, CortexOutputModel)
            assert len(result.actions) == 1
            mock_convert.assert_called_once()

    @pytest.mark.asyncio
    async def test_ask_without_function_calls(self, glm4_llm):
        """Test ask method when model returns no function calls."""
        mock_message = MagicMock()
        mock_message.tool_calls = None

        mock_choice = MagicMock()
        mock_choice.message = mock_message

        mock_response = MagicMock()
        mock_response.choices = [mock_choice]

        glm4_llm._client.chat.completions.create = AsyncMock(return_value=mock_response)

        result = await glm4_llm.ask("Test prompt")

        assert result is None

    @pytest.mark.asyncio
    async def test_ask_formats_messages_correctly(self, glm4_llm):
        """Test that ask method formats messages and calls API correctly."""
        mock_message = MagicMock()
        mock_message.tool_calls = None

        mock_choice = MagicMock()
        mock_choice.message = mock_message

        mock_response = MagicMock()
        mock_response.choices = [mock_choice]

        glm4_llm._client.chat.completions.create = AsyncMock(return_value=mock_response)

        await glm4_llm.ask("Test prompt")

        glm4_llm._client.chat.completions.create.assert_called_once()
        call_args = glm4_llm._client.chat.completions.create.call_args
        assert call_args.kwargs["model"] == "glm-4.7"
        assert "messages" in call_args.kwargs

    @pytest.mark.asyncio
    async def test_ask_handles_api_error(self, glm4_llm):
        """Test that API errors are handled gracefully."""
        glm4_llm._client.chat.completions.create = AsyncMock(
            side_effect=Exception("API Error")
        )

        result = await glm4_llm.ask("Test prompt")

        assert result is None

    @pytest.mark.asyncio
    async def test_ask_sets_io_provider_times(self, glm4_llm):
        """Test that IO provider times are set correctly."""
        mock_message = MagicMock()
        mock_message.tool_calls = None

        mock_choice = MagicMock()
        mock_choice.message = mock_message

        mock_response = MagicMock()
        mock_response.choices = [mock_choice]

        glm4_llm._client.chat.completions.create = AsyncMock(return_value=mock_response)

        await glm4_llm.ask("Test prompt")

        assert glm4_llm.io_provider.llm_start_time is not None
        assert glm4_llm.io_provider.llm_end_time is not None
