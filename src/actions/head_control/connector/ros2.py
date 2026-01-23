import logging

from actions.base import ActionConfig, ActionConnector
from actions.head_control.interface import HeadInput


class HeadControlRos2Connector(ActionConnector[ActionConfig, HeadInput]):
    """
    Connector to link HeadControl action with ROS2.
    """

    def __init__(self, config: ActionConfig):
        """
        Initialize the HeadControlRos2Connector with the given configuration.

        Parameters
        ----------
        config : ActionConfig
            Configuration parameters for the connector.
        """
        super().__init__(config)
        logging.info("HeadControl ROS2 connector initialized")

    async def connect(self, output_interface: HeadInput) -> None:
        """
        Connect to the ROS2 system and send the appropriate head command.

        Parameters
        ----------
        output_interface : HeadInput
            The head input containing the action to be performed.
        """
        new_msg = {"head": ""}

        if output_interface.action == "look left":
            new_msg["head"] = "look_left"
        elif output_interface.action == "look right":
            new_msg["head"] = "look_right"
        elif output_interface.action == "look up":
            new_msg["head"] = "look_up"
        elif output_interface.action == "look down":
            new_msg["head"] = "look_down"
        elif output_interface.action == "look at person":
            new_msg["head"] = "look_at_person"
        elif output_interface.action == "center":
            new_msg["head"] = "center"
        else:
            logging.info(f"Unknown head action: {output_interface.action}")
            return

        logging.info(f"SendThisToROS2: {new_msg}")
