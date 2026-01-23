from dataclasses import dataclass
from enum import Enum
from typing import Optional

from actions.base import Interface


class GoalPriority(str, Enum):
    """Priority level for robot goals."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class SetGoalInput:
    """
    Input payload for setting a behavioral goal for the robot.

    The 'action' field contains the goal description (e.g., "guard the house",
    "patrol the perimeter", "find my owner").
    The 'priority' field indicates the importance of this goal.
    The 'description' field provides additional context about the goal.

    Examples
    --------
    - User says: "Your goal is to guard the house" -> action = "guard the house"
    - User says: "Set a high priority goal to find my keys" -> action = "find my keys", priority = "high"
    - User says: "Focus on patrolling the yard" -> action = "patrol the yard"
    """

    action: str
    priority: GoalPriority = GoalPriority.MEDIUM
    description: Optional[str] = ""


@dataclass
class SetGoal(Interface[SetGoalInput, SetGoalInput]):
    """
    Set a behavioral goal or objective for the robot.

    This action allows setting high-level goals that guide the robot's behavior.
    Goals can have different priorities and optional descriptions.
    The robot will work toward achieving these goals based on priority.

    The 'action' field should contain the goal description.
    Extract the goal from user commands like "your goal is [goal]" or "focus on [goal]".
    """

    input: SetGoalInput
    output: SetGoalInput
