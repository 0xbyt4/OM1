from unittest.mock import Mock, patch

import pytest

from actions.set_goal.connector.goal import SetGoalConfig, SetGoalConnector
from actions.set_goal.interface import GoalPriority, SetGoalInput


@pytest.fixture
def mock_goal_provider():
    """Mock GoalProvider."""
    with patch("actions.set_goal.connector.goal.GoalProvider") as mock:
        mock_instance = Mock()
        mock_instance.set_goal.return_value = Mock(
            name="test goal", priority="medium", status="active"
        )
        mock.return_value = mock_instance
        yield mock_instance


@pytest.fixture
def goal_connector(mock_goal_provider):
    """Create SetGoalConnector with mocked dependencies."""
    config = SetGoalConfig(
        base_url="",
        timeout=5,
        refresh_interval=30,
    )
    connector = SetGoalConnector(config)
    connector.goal_provider = mock_goal_provider
    return connector


class TestSetGoalConfig:
    """Tests for SetGoalConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        config = SetGoalConfig()
        assert config.base_url == ""
        assert config.timeout == 5
        assert config.refresh_interval == 30

    def test_custom_config(self):
        """Test custom configuration values."""
        config = SetGoalConfig(
            base_url="http://localhost:5000/goals",
            timeout=10,
            refresh_interval=60,
        )
        assert config.base_url == "http://localhost:5000/goals"
        assert config.timeout == 10
        assert config.refresh_interval == 60


class TestSetGoalInput:
    """Tests for SetGoalInput."""

    def test_default_values(self):
        """Test default input values."""
        input_data = SetGoalInput(action="guard the house")
        assert input_data.action == "guard the house"
        assert input_data.priority == GoalPriority.MEDIUM
        assert input_data.description == ""

    def test_custom_values(self):
        """Test custom input values."""
        input_data = SetGoalInput(
            action="find my owner",
            priority=GoalPriority.HIGH,
            description="Search all rooms",
        )
        assert input_data.action == "find my owner"
        assert input_data.priority == GoalPriority.HIGH
        assert input_data.description == "Search all rooms"


class TestSetGoalConnector:
    """Tests for SetGoalConnector."""

    @pytest.mark.asyncio
    async def test_connect_basic_goal(self, goal_connector):
        """Test connecting with a basic goal."""
        input_data = SetGoalInput(action="guard the house")

        await goal_connector.connect(input_data)

        goal_connector.goal_provider.set_goal.assert_called_once_with(
            name="guard the house",
            priority="medium",
            description="",
        )

    @pytest.mark.asyncio
    async def test_connect_goal_with_priority(self, goal_connector):
        """Test connecting with a prioritized goal."""
        input_data = SetGoalInput(
            action="find my owner",
            priority=GoalPriority.HIGH,
        )

        await goal_connector.connect(input_data)

        goal_connector.goal_provider.set_goal.assert_called_once_with(
            name="find my owner",
            priority="high",
            description="",
        )

    @pytest.mark.asyncio
    async def test_connect_goal_with_description(self, goal_connector):
        """Test connecting with a goal with description."""
        input_data = SetGoalInput(
            action="patrol the yard",
            priority=GoalPriority.CRITICAL,
            description="Check the perimeter every 10 minutes",
        )

        await goal_connector.connect(input_data)

        goal_connector.goal_provider.set_goal.assert_called_once_with(
            name="patrol the yard",
            priority="critical",
            description="Check the perimeter every 10 minutes",
        )

    @pytest.mark.asyncio
    async def test_connect_empty_action(self, goal_connector):
        """Test connecting with an empty action."""
        input_data = SetGoalInput(action="")

        await goal_connector.connect(input_data)

        goal_connector.goal_provider.set_goal.assert_not_called()

    @pytest.mark.asyncio
    async def test_connect_whitespace_action(self, goal_connector):
        """Test connecting with whitespace-only action."""
        input_data = SetGoalInput(action="   ")

        await goal_connector.connect(input_data)

        goal_connector.goal_provider.set_goal.assert_not_called()

    @pytest.mark.asyncio
    async def test_connect_strips_whitespace(self, goal_connector):
        """Test that action is stripped of whitespace."""
        input_data = SetGoalInput(action="  guard the house  ")

        await goal_connector.connect(input_data)

        goal_connector.goal_provider.set_goal.assert_called_once_with(
            name="guard the house",
            priority="medium",
            description="",
        )

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "priority,expected",
        [
            (GoalPriority.LOW, "low"),
            (GoalPriority.MEDIUM, "medium"),
            (GoalPriority.HIGH, "high"),
            (GoalPriority.CRITICAL, "critical"),
        ],
    )
    async def test_connect_all_priorities(self, goal_connector, priority, expected):
        """Test connecting with all priority levels."""
        input_data = SetGoalInput(action="test goal", priority=priority)

        await goal_connector.connect(input_data)

        call_args = goal_connector.goal_provider.set_goal.call_args
        assert call_args.kwargs["priority"] == expected


class TestGoalPriorityEnum:
    """Tests for GoalPriority enum."""

    def test_enum_values(self):
        """Test that enum values are correct strings."""
        assert GoalPriority.LOW.value == "low"
        assert GoalPriority.MEDIUM.value == "medium"
        assert GoalPriority.HIGH.value == "high"
        assert GoalPriority.CRITICAL.value == "critical"

    def test_enum_is_str(self):
        """Test that enum is a string enum."""
        assert isinstance(GoalPriority.LOW, str)
        assert GoalPriority.LOW == "low"
