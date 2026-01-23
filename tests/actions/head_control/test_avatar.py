from unittest.mock import Mock, patch

import pytest

from actions.base import ActionConfig
from actions.head_control.connector.avatar import HeadControlAvatarConnector
from actions.head_control.interface import HeadAction, HeadInput


@pytest.fixture
def mock_avatar_provider():
    """Mock AvatarProvider."""
    with patch(
        "actions.head_control.connector.avatar.AvatarProvider"
    ) as mock_provider_class:
        mock_instance = Mock()
        mock_provider_class.return_value = mock_instance
        yield mock_instance


@pytest.fixture
def avatar_connector(mock_avatar_provider):
    """Create HeadControlAvatarConnector with mocked AvatarProvider."""
    config = ActionConfig()
    return HeadControlAvatarConnector(config)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "action,expected_command",
    [
        (HeadAction.LOOK_LEFT, "LookLeft"),
        (HeadAction.LOOK_RIGHT, "LookRight"),
        (HeadAction.LOOK_UP, "LookUp"),
        (HeadAction.LOOK_DOWN, "LookDown"),
        (HeadAction.LOOK_AT_PERSON, "LookAtPerson"),
        (HeadAction.CENTER, "Center"),
    ],
)
async def test_head_actions_send_avatar_command(
    avatar_connector, mock_avatar_provider, action, expected_command
):
    """Test that head actions send correct commands to AvatarProvider."""
    input_interface = HeadInput(action=action)

    await avatar_connector.connect(input_interface)

    mock_avatar_provider.send_avatar_command.assert_called_with(expected_command)


@pytest.mark.asyncio
async def test_unknown_action_logs_warning(avatar_connector, mock_avatar_provider):
    """Test that unknown actions are logged as warnings."""
    with patch("actions.head_control.connector.avatar.logging") as mock_logging:
        input_interface = HeadInput(action="unknown_action")

        await avatar_connector.connect(input_interface)

        mock_logging.warning.assert_called_with("Unknown head action: unknown_action")
        mock_avatar_provider.send_avatar_command.assert_not_called()


def test_connector_initialization(mock_avatar_provider):
    """Test that connector initializes with AvatarProvider."""
    with patch("actions.head_control.connector.avatar.logging") as mock_logging:
        config = ActionConfig()
        connector = HeadControlAvatarConnector(config)

        assert connector is not None
        assert connector.avatar_provider is not None
        mock_logging.info.assert_called_with(
            "HeadControl system initiated with AvatarProvider"
        )


def test_stop_calls_avatar_provider_stop(avatar_connector, mock_avatar_provider):
    """Test that stop() calls AvatarProvider.stop()."""
    with patch("actions.head_control.connector.avatar.logging") as mock_logging:
        avatar_connector.stop()

        mock_avatar_provider.stop.assert_called_once()
        mock_logging.info.assert_called_with("HeadControl AvatarProvider stopped")
