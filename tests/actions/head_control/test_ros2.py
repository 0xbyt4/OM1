from unittest.mock import patch

import pytest

from actions.base import ActionConfig
from actions.head_control.connector.ros2 import HeadControlRos2Connector
from actions.head_control.interface import HeadAction, HeadInput


@pytest.fixture
def ros2_connector():
    """Create HeadControlRos2Connector with default config."""
    config = ActionConfig()
    return HeadControlRos2Connector(config)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "action,expected_msg",
    [
        (HeadAction.LOOK_LEFT, "look_left"),
        (HeadAction.LOOK_RIGHT, "look_right"),
        (HeadAction.LOOK_UP, "look_up"),
        (HeadAction.LOOK_DOWN, "look_down"),
        (HeadAction.LOOK_AT_PERSON, "look_at_person"),
        (HeadAction.CENTER, "center"),
    ],
)
async def test_head_actions(ros2_connector, action, expected_msg):
    """Test that head actions are correctly mapped to ROS2 messages."""
    with patch("actions.head_control.connector.ros2.logging") as mock_logging:
        input_interface = HeadInput(action=action)

        await ros2_connector.connect(input_interface)

        mock_logging.info.assert_called_with(
            f"SendThisToROS2: {{'head': '{expected_msg}'}}"
        )


@pytest.mark.asyncio
async def test_unknown_action_logs_info(ros2_connector):
    """Test that unknown actions are logged."""
    with patch("actions.head_control.connector.ros2.logging") as mock_logging:
        input_interface = HeadInput(action="unknown_action")

        await ros2_connector.connect(input_interface)

        mock_logging.info.assert_called_with("Unknown head action: unknown_action")


def test_connector_initialization():
    """Test that connector initializes correctly."""
    with patch("actions.head_control.connector.ros2.logging") as mock_logging:
        config = ActionConfig()
        connector = HeadControlRos2Connector(config)

        assert connector is not None
        mock_logging.info.assert_called_with("HeadControl ROS2 connector initialized")


def test_head_action_enum_values():
    """Test that HeadAction enum has expected values."""
    assert HeadAction.LOOK_LEFT.value == "look left"
    assert HeadAction.LOOK_RIGHT.value == "look right"
    assert HeadAction.LOOK_UP.value == "look up"
    assert HeadAction.LOOK_DOWN.value == "look down"
    assert HeadAction.LOOK_AT_PERSON.value == "look at person"
    assert HeadAction.CENTER.value == "center"


def test_head_input_creation():
    """Test that HeadInput can be created with HeadAction."""
    input_data = HeadInput(action=HeadAction.LOOK_LEFT)

    assert input_data.action == HeadAction.LOOK_LEFT
