import logging

from actions.base import ActionConfig, ActionConnector
from actions.head_control.interface import HeadInput
from providers.avatar_provider import AvatarProvider


class HeadControlAvatarConnector(ActionConnector[ActionConfig, HeadInput]):
    """
    Connector to link HeadControl action with AvatarProvider.
    """

    def __init__(self, config: ActionConfig):
        """
        Initialize the HeadControlAvatarConnector with AvatarProvider.

        Parameters
        ----------
        config : ActionConfig
            Configuration parameters for the connector.
        """
        super().__init__(config)
        self.avatar_provider = AvatarProvider()
        logging.info("HeadControl system initiated with AvatarProvider")

    async def connect(self, output_interface: HeadInput) -> None:
        """
        Send head command via AvatarProvider.

        Parameters
        ----------
        output_interface : HeadInput
            The head input containing the action to be performed.
        """
        if output_interface.action == "look left":
            self.avatar_provider.send_avatar_command("LookLeft")
        elif output_interface.action == "look right":
            self.avatar_provider.send_avatar_command("LookRight")
        elif output_interface.action == "look up":
            self.avatar_provider.send_avatar_command("LookUp")
        elif output_interface.action == "look down":
            self.avatar_provider.send_avatar_command("LookDown")
        elif output_interface.action == "look at person":
            self.avatar_provider.send_avatar_command("LookAtPerson")
        elif output_interface.action == "center":
            self.avatar_provider.send_avatar_command("Center")
        else:
            logging.warning(f"Unknown head action: {output_interface.action}")
            return

        logging.info(f"Avatar head command sent: {output_interface.action}")

    def stop(self):
        """
        Stop and cleanup AvatarProvider.
        """
        self.avatar_provider.stop()
        logging.info("HeadControl AvatarProvider stopped")
