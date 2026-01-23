import logging

from pydantic import Field

from actions.base import ActionConfig, ActionConnector
from actions.set_goal.interface import SetGoalInput
from providers.goal_provider import GoalProvider


class SetGoalConfig(ActionConfig):
    """
    Configuration for SetGoal connector.

    Parameters
    ----------
    base_url : str
        The base URL for the goals API. If empty, goals are stored locally only.
    timeout : int
        Timeout for the HTTP requests in seconds.
    refresh_interval : int
        Interval to refresh the goals list in seconds.
    """

    base_url: str = Field(
        default="",
        description="The base URL for the goals API. If empty, goals are stored locally.",
    )
    timeout: int = Field(
        default=5,
        description="Timeout for the HTTP requests in seconds.",
    )
    refresh_interval: int = Field(
        default=30,
        description="Interval to refresh the goals list in seconds.",
    )


class SetGoalConnector(ActionConnector[SetGoalConfig, SetGoalInput]):
    """
    Connector that sets behavioral goals for the robot.

    Goals can be persisted via an HTTP API (if configured) or stored locally in memory.
    """

    def __init__(self, config: SetGoalConfig):
        """
        Initialize the SetGoalConnector.

        Parameters
        ----------
        config : SetGoalConfig
            Configuration for the action connector.
        """
        super().__init__(config)

        self.goal_provider = GoalProvider(
            base_url=self.config.base_url,
            timeout=self.config.timeout,
            refresh_interval=self.config.refresh_interval,
        )
        self.goal_provider.start()

    async def connect(self, output_interface: SetGoalInput) -> None:
        """
        Connect the input protocol to the set goal action.

        Parameters
        ----------
        output_interface : SetGoalInput
            The input protocol containing the goal details.
        """
        action = output_interface.action.strip()
        if not action:
            logging.warning("SetGoal received empty action")
            return

        priority = output_interface.priority.value
        description = output_interface.description or ""

        goal = self.goal_provider.set_goal(
            name=action,
            priority=priority,
            description=description,
        )

        logging.info(f"SetGoal: Goal '{goal.name}' set with priority '{goal.priority}'")
