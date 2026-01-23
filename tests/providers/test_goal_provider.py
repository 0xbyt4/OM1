from unittest.mock import patch

import pytest

from providers.goal_provider import Goal, GoalProvider


@pytest.fixture
def goal_provider():
    """Create a GoalProvider instance for testing (local mode)."""
    GoalProvider.reset()
    provider = GoalProvider(base_url="", timeout=5, refresh_interval=30)
    yield provider
    provider.stop()
    GoalProvider.reset()


@pytest.fixture
def goal_provider_with_api():
    """Create a GoalProvider instance with API configured."""
    GoalProvider.reset()
    provider = GoalProvider(
        base_url="http://localhost:5000/goals",
        timeout=5,
        refresh_interval=30,
    )
    yield provider
    provider.stop()
    GoalProvider.reset()


class TestGoal:
    """Tests for the Goal dataclass."""

    def test_goal_default_values(self):
        """Test that Goal has correct default values."""
        goal = Goal(name="test goal")
        assert goal.name == "test goal"
        assert goal.priority == "medium"
        assert goal.description == ""
        assert goal.status == "active"
        assert goal.created_at != ""

    def test_goal_custom_values(self):
        """Test Goal with custom values."""
        goal = Goal(
            name="guard the house",
            priority="high",
            description="Watch for intruders",
            status="active",
        )
        assert goal.name == "guard the house"
        assert goal.priority == "high"
        assert goal.description == "Watch for intruders"


class TestGoalProvider:
    """Tests for the GoalProvider class."""

    def test_set_goal_basic(self, goal_provider):
        """Test setting a basic goal."""
        goal = goal_provider.set_goal(name="guard the house")
        assert goal.name == "guard the house"
        assert goal.priority == "medium"
        assert goal.status == "active"

    def test_set_goal_with_priority(self, goal_provider):
        """Test setting a goal with priority."""
        goal = goal_provider.set_goal(
            name="find my owner",
            priority="high",
            description="Search all rooms",
        )
        assert goal.name == "find my owner"
        assert goal.priority == "high"
        assert goal.description == "Search all rooms"

    def test_get_goal(self, goal_provider):
        """Test retrieving a goal by name."""
        goal_provider.set_goal(name="patrol the yard")
        goal = goal_provider.get_goal("patrol the yard")
        assert goal is not None
        assert goal.name == "patrol the yard"

    def test_get_goal_case_insensitive(self, goal_provider):
        """Test that goal retrieval is case insensitive."""
        goal_provider.set_goal(name="Guard The House")
        goal = goal_provider.get_goal("guard the house")
        assert goal is not None
        assert goal.name == "Guard The House"

    def test_get_goal_not_found(self, goal_provider):
        """Test retrieving a non-existent goal."""
        goal = goal_provider.get_goal("non-existent")
        assert goal is None

    def test_get_all_goals(self, goal_provider):
        """Test retrieving all goals."""
        goal_provider.set_goal(name="goal 1")
        goal_provider.set_goal(name="goal 2")
        goal_provider.set_goal(name="goal 3")
        all_goals = goal_provider.get_all_goals()
        assert len(all_goals) == 3

    def test_get_active_goals_sorted(self, goal_provider):
        """Test that active goals are sorted by priority."""
        goal_provider.set_goal(name="low priority", priority="low")
        goal_provider.set_goal(name="critical", priority="critical")
        goal_provider.set_goal(name="medium", priority="medium")
        goal_provider.set_goal(name="high", priority="high")

        active = goal_provider.get_active_goals()
        assert len(active) == 4
        assert active[0].priority == "critical"
        assert active[1].priority == "high"
        assert active[2].priority == "medium"
        assert active[3].priority == "low"

    def test_get_current_goal(self, goal_provider):
        """Test getting the current highest priority goal."""
        goal_provider.set_goal(name="low priority", priority="low")
        goal_provider.set_goal(name="high priority", priority="high")

        current = goal_provider.get_current_goal()
        assert current is not None
        assert current.name == "high priority"

    def test_complete_goal(self, goal_provider):
        """Test completing a goal."""
        goal_provider.set_goal(name="test goal")
        result = goal_provider.complete_goal("test goal")
        assert result is True

        goal = goal_provider.get_goal("test goal")
        assert goal.status == "completed"

    def test_complete_nonexistent_goal(self, goal_provider):
        """Test completing a non-existent goal."""
        result = goal_provider.complete_goal("non-existent")
        assert result is False

    def test_cancel_goal(self, goal_provider):
        """Test cancelling a goal."""
        goal_provider.set_goal(name="test goal")
        result = goal_provider.cancel_goal("test goal")
        assert result is True

        goal = goal_provider.get_goal("test goal")
        assert goal.status == "cancelled"

    def test_clear_all_goals(self, goal_provider):
        """Test clearing all goals."""
        goal_provider.set_goal(name="goal 1")
        goal_provider.set_goal(name="goal 2")
        goal_provider.clear_all_goals()

        all_goals = goal_provider.get_all_goals()
        assert len(all_goals) == 0

    def test_to_prompt_context_no_goals(self, goal_provider):
        """Test prompt context with no goals."""
        context = goal_provider.to_prompt_context()
        assert context == "You have no active goals."

    def test_to_prompt_context_with_goals(self, goal_provider):
        """Test prompt context with active goals."""
        goal_provider.set_goal(name="guard the house", priority="high")
        goal_provider.set_goal(
            name="find my owner",
            priority="critical",
            description="Search all rooms",
        )

        context = goal_provider.to_prompt_context()
        assert "Your current goals" in context
        assert "[CRITICAL] find my owner" in context
        assert "[HIGH] guard the house" in context

    def test_completed_goals_not_in_active(self, goal_provider):
        """Test that completed goals are not in active list."""
        goal_provider.set_goal(name="active goal")
        goal_provider.set_goal(name="completed goal")
        goal_provider.complete_goal("completed goal")

        active = goal_provider.get_active_goals()
        assert len(active) == 1
        assert active[0].name == "active goal"


class TestGoalProviderWithAPI:
    """Tests for GoalProvider with API integration."""

    @patch("providers.goal_provider.requests.post")
    def test_persist_goal_to_api(self, mock_post, goal_provider_with_api):
        """Test that goals are persisted to API."""
        mock_post.return_value.status_code = 200

        goal_provider_with_api.set_goal(name="test goal")

        mock_post.assert_called_once()
        call_args = mock_post.call_args
        assert "test goal" in str(call_args)

    @patch("providers.goal_provider.requests.post")
    def test_persist_goal_api_error(self, mock_post, goal_provider_with_api):
        """Test handling of API errors when persisting goals."""
        mock_post.return_value.status_code = 500

        goal = goal_provider_with_api.set_goal(name="test goal")

        assert goal is not None
        assert goal.name == "test goal"

    @patch("providers.goal_provider.requests.get")
    def test_fetch_goals_from_api(self, mock_get, goal_provider_with_api):
        """Test fetching goals from API."""
        mock_get.return_value.status_code = 200
        mock_get.return_value.json.return_value = {
            "goals": [
                {"name": "api goal 1", "priority": "high"},
                {"name": "api goal 2", "priority": "low"},
            ]
        }

        goal_provider_with_api._fetch()

        all_goals = goal_provider_with_api.get_all_goals()
        assert len(all_goals) == 2
