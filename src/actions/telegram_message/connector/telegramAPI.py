import logging
import os

import aiohttp
from dotenv import load_dotenv

from actions.base import ActionConfig, ActionConnector
from actions.telegram_message.interface import TelegramMessageInput


class TelegramAPIConnector(ActionConnector[ActionConfig, TelegramMessageInput]):
    """
    Connector for Telegram Bot API.

    This connector integrates with Telegram Bot API to send messages from the robot.
    """

    def __init__(self, config: ActionConfig):
        """
        Initialize the Telegram API connector.

        Parameters
        ----------
        config : ActionConfig
            Configuration for the action connector.
        """
        super().__init__(config)

        load_dotenv()

        self.bot_token = os.getenv("TELEGRAM_BOT_TOKEN")
        self.chat_id = os.getenv("TELEGRAM_CHAT_ID")

        if not self.bot_token:
            logging.warning("TELEGRAM_BOT_TOKEN not set in environment")
        if not self.chat_id:
            logging.warning("TELEGRAM_CHAT_ID not set in environment")

    async def connect(self, output_interface: TelegramMessageInput) -> None:
        """
        Send message via Telegram Bot API.

        Parameters
        ----------
        output_interface : TelegramMessageInput
            The TelegramMessageInput interface containing the message text.
        """
        if not self.bot_token or not self.chat_id:
            logging.error("Telegram credentials not configured")
            return

        try:
            message_text = output_interface.action
            logging.info(f"SendThisToTelegram: {message_text}")

            url = f"https://api.telegram.org/bot{self.bot_token}/sendMessage"
            payload = {
                "chat_id": self.chat_id,
                "text": message_text,
                "parse_mode": "HTML",
            }

            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=payload) as response:
                    if response.status == 200:
                        data = await response.json()
                        message_id = data.get("result", {}).get("message_id")
                        logging.info(
                            f"Telegram message sent successfully! Message ID: {message_id}"
                        )
                    else:
                        error_text = await response.text()
                        logging.error(
                            f"Telegram API error: {response.status} - {error_text}"
                        )

        except Exception as e:
            logging.error(f"Failed to send Telegram message: {str(e)}")
            raise
