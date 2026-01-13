"""Tests for Telegram Message action."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from actions.base import ActionConfig
from actions.telegram_message.connector.telegramAPI import TelegramAPIConnector
from actions.telegram_message.interface import TelegramMessage, TelegramMessageInput


class TestTelegramMessageInterface:
    """Tests for TelegramMessageInput interface."""

    def test_telegram_message_input_default(self):
        """Test TelegramMessageInput with default values."""
        input_obj = TelegramMessageInput()
        assert input_obj.action == ""

    def test_telegram_message_input_with_value(self):
        """Test TelegramMessageInput with custom value."""
        input_obj = TelegramMessageInput(action="Hello from robot!")
        assert input_obj.action == "Hello from robot!"

    def test_telegram_message_input_with_emoji(self):
        """Test TelegramMessageInput with emoji."""
        input_obj = TelegramMessageInput(action="Battery low! Please charge.")
        assert "Battery" in input_obj.action

    def test_telegram_message_interface(self):
        """Test TelegramMessage interface creation."""
        input_obj = TelegramMessageInput(action="Test message")
        output_obj = TelegramMessageInput(action="Test message")
        message = TelegramMessage(input=input_obj, output=output_obj)
        assert message.input.action == "Test message"
        assert message.output.action == "Test message"


class TestTelegramAPIConnectorInit:
    """Tests for TelegramAPIConnector initialization."""

    def test_init_with_credentials(self):
        """Test initialization with credentials."""
        with (
            patch("actions.telegram_message.connector.telegramAPI.load_dotenv"),
            patch.dict(
                "os.environ",
                {
                    "TELEGRAM_BOT_TOKEN": "test-bot-token",
                    "TELEGRAM_CHAT_ID": "test-chat-id",
                },
                clear=True,
            ),
        ):
            config = ActionConfig()
            connector = TelegramAPIConnector(config)
            assert connector.bot_token == "test-bot-token"
            assert connector.chat_id == "test-chat-id"

    def test_init_without_bot_token(self):
        """Test initialization without bot token logs warning."""
        with (
            patch("actions.telegram_message.connector.telegramAPI.load_dotenv"),
            patch.dict("os.environ", {"TELEGRAM_CHAT_ID": "test-chat-id"}, clear=True),
            patch(
                "actions.telegram_message.connector.telegramAPI.logging.warning"
            ) as mock_warning,
        ):
            config = ActionConfig()
            connector = TelegramAPIConnector(config)
            assert connector.bot_token is None
            mock_warning.assert_any_call("TELEGRAM_BOT_TOKEN not set in environment")

    def test_init_without_chat_id(self):
        """Test initialization without chat id logs warning."""
        with (
            patch("actions.telegram_message.connector.telegramAPI.load_dotenv"),
            patch.dict("os.environ", {"TELEGRAM_BOT_TOKEN": "test-token"}, clear=True),
            patch(
                "actions.telegram_message.connector.telegramAPI.logging.warning"
            ) as mock_warning,
        ):
            config = ActionConfig()
            connector = TelegramAPIConnector(config)
            assert connector.chat_id is None
            mock_warning.assert_any_call("TELEGRAM_CHAT_ID not set in environment")

    def test_init_without_any_credentials(self):
        """Test initialization without any credentials logs both warnings."""
        with (
            patch("actions.telegram_message.connector.telegramAPI.load_dotenv"),
            patch.dict("os.environ", {}, clear=True),
            patch(
                "actions.telegram_message.connector.telegramAPI.logging.warning"
            ) as mock_warning,
        ):
            config = ActionConfig()
            connector = TelegramAPIConnector(config)
            assert connector.bot_token is None
            assert connector.chat_id is None
            assert mock_warning.call_count >= 2


class TestTelegramAPIConnectorConnect:
    """Tests for TelegramAPIConnector.connect method."""

    @pytest.fixture
    def connector_with_credentials(self):
        """Create a connector with mocked credentials."""
        with (
            patch("actions.telegram_message.connector.telegramAPI.load_dotenv"),
            patch.dict(
                "os.environ",
                {
                    "TELEGRAM_BOT_TOKEN": "test-bot-token",
                    "TELEGRAM_CHAT_ID": "123456789",
                },
                clear=True,
            ),
        ):
            config = ActionConfig()
            return TelegramAPIConnector(config)

    @pytest.mark.asyncio
    async def test_connect_without_credentials_returns_early(self):
        """Test that connect returns early without credentials."""
        with (
            patch("actions.telegram_message.connector.telegramAPI.load_dotenv"),
            patch.dict("os.environ", {}, clear=True),
        ):
            config = ActionConfig()
            connector = TelegramAPIConnector(config)

            with patch(
                "actions.telegram_message.connector.telegramAPI.logging.error"
            ) as mock_error:
                input_obj = TelegramMessageInput(action="Test")
                await connector.connect(input_obj)
                mock_error.assert_called_with("Telegram credentials not configured")

    @pytest.mark.asyncio
    async def test_connect_logs_message(self, connector_with_credentials):
        """Test that connect logs the message being sent."""
        with patch(
            "actions.telegram_message.connector.telegramAPI.aiohttp.ClientSession"
        ) as mock_session:
            mock_response = AsyncMock()
            mock_response.status = 200
            mock_response.json = AsyncMock(
                return_value={"result": {"message_id": 12345}}
            )

            mock_post = AsyncMock(return_value=mock_response)
            mock_post.__aenter__ = AsyncMock(return_value=mock_response)
            mock_post.__aexit__ = AsyncMock(return_value=None)

            mock_session_instance = MagicMock()
            mock_session_instance.post = MagicMock(return_value=mock_post)
            mock_session_instance.__aenter__ = AsyncMock(
                return_value=mock_session_instance
            )
            mock_session_instance.__aexit__ = AsyncMock(return_value=None)

            mock_session.return_value = mock_session_instance

            with patch(
                "actions.telegram_message.connector.telegramAPI.logging.info"
            ) as mock_info:
                input_obj = TelegramMessageInput(action="Test notification")
                await connector_with_credentials.connect(input_obj)
                mock_info.assert_any_call("SendThisToTelegram: Test notification")

    @pytest.mark.asyncio
    async def test_connect_uses_correct_api_url(self, connector_with_credentials):
        """Test that connect calls correct Telegram API URL."""
        with patch(
            "actions.telegram_message.connector.telegramAPI.aiohttp.ClientSession"
        ) as mock_session:
            mock_response = AsyncMock()
            mock_response.status = 200
            mock_response.json = AsyncMock(
                return_value={"result": {"message_id": 12345}}
            )

            mock_post = AsyncMock(return_value=mock_response)
            mock_post.__aenter__ = AsyncMock(return_value=mock_response)
            mock_post.__aexit__ = AsyncMock(return_value=None)

            mock_session_instance = MagicMock()
            mock_session_instance.post = MagicMock(return_value=mock_post)
            mock_session_instance.__aenter__ = AsyncMock(
                return_value=mock_session_instance
            )
            mock_session_instance.__aexit__ = AsyncMock(return_value=None)

            mock_session.return_value = mock_session_instance

            input_obj = TelegramMessageInput(action="Test")
            await connector_with_credentials.connect(input_obj)

            mock_session_instance.post.assert_called_once()
            call_args = mock_session_instance.post.call_args
            assert "api.telegram.org" in call_args[0][0]
            assert "test-bot-token" in call_args[0][0]


class TestActionLoading:
    """Tests for action loading mechanism."""

    def test_action_loads_successfully(self):
        """Test that telegram_message action loads via load_action."""
        import sys

        sys.path.insert(0, "src")
        from actions import load_action

        config: dict[str, str | dict[str, str]] = {
            "name": "telegram_message",
            "llm_label": "notify",
            "connector": "telegramAPI",
        }

        with (
            patch("actions.telegram_message.connector.telegramAPI.load_dotenv"),
            patch.dict(
                "os.environ",
                {
                    "TELEGRAM_BOT_TOKEN": "test-token",
                    "TELEGRAM_CHAT_ID": "test-chat",
                },
                clear=True,
            ),
        ):
            action = load_action(config)
            assert action.name == "telegram_message"
            assert action.llm_label == "notify"

    def test_action_description_generated(self):
        """Test that action description is generated for LLM."""
        import sys

        sys.path.insert(0, "src")
        from actions import describe_action

        desc = describe_action("telegram_message", "notify", False)
        assert desc is not None
        assert "NOTIFY" in desc
        assert "Telegram" in desc
        assert "type=notify" in desc
