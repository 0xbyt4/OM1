import json
import logging

import requests


class UbTtsProvider:
    """
    Provider for the Ubtech Text-to-Speech (TTS) service.
    """

    def __init__(self, url: str):
        """
        Initialize the Ubtech TTS Provider.

        Parameters
        ----------
        url : str
            The URL for the Ubtech TTS service.
        """
        self.tts_url = url
        self.headers = {"Content-Type": "application/json"}
        logging.info(f"Ubtech TTS Provider initialized for URL: {self.tts_url}")

    def speak(self, tts: str, interrupt: bool = True, timestamp: int = 0) -> bool:
        """
        Sends a request to the TTS service.

        Parameters
        ----------
        tts : str
            The text to be converted to speech.
        interrupt : bool
            Whether to interrupt any currently playing audio. Defaults to True.
        timestamp : int
            Timestamp identifier for the TTS request. Defaults to 0.

        Returns
        -------
        bool
            True on success, False otherwise.
        """
        payload = {"tts": tts, "interrupt": interrupt, "timestamp": timestamp}
        try:
            response = requests.put(
                url=self.tts_url,
                data=json.dumps(payload),
                headers=self.headers,
                timeout=5,
            )
            response.raise_for_status()
            res = response.json()
            return res.get("code") == 0
        except requests.exceptions.RequestException as e:
            logging.error(f"Failed to send TTS command: {e}")
            return False

    def get_tts_status(self, timestamp: int) -> str:
        """
        Gets the status of a specific TTS task.

        Parameters
        ----------
        timestamp : int
            Timestamp identifier for the TTS task to check.

        Returns
        -------
        str
            Status of the TTS task. Possible values: 'build', 'wait', 'run', 'idle', 'error'.
        """
        try:
            params = {"timestamp": timestamp}
            response = requests.get(
                url=self.tts_url, headers=self.headers, params=params, timeout=2
            )
            res = response.json()
            if res.get("code") == 0:
                return res.get("status", "error")
            return "error"
        except requests.exceptions.RequestException:
            return "error"
