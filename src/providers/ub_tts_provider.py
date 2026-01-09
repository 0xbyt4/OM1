import json
import logging

import requests

from .health_monitor_provider import HealthMonitorProvider


class UbTtsProvider:
    """
    Provider for the Ubtech Text-to-Speech (TTS) service.
    """

    def __init__(self, url: str):
        self.tts_url = url
        self.headers = {"Content-Type": "application/json"}
        self._health_monitor = HealthMonitorProvider()
        self._health_monitor.register(
            "UbTtsProvider",
            metadata={"type": "tts"},
            recovery_callback=self._recover,
        )
        logging.info(f"Ubtech TTS Provider initialized for URL: {self.tts_url}")

    def _recover(self) -> bool:
        """
        Attempt to recover by testing connectivity to the TTS service.

        Returns
        -------
        bool
            True if service is reachable, False otherwise.
        """
        try:
            logging.info("UbTtsProvider: Attempting recovery...")
            response = requests.get(
                url=self.tts_url,
                headers=self.headers,
                params={"timestamp": 0},
                timeout=5,
            )
            if response.status_code == 200:
                self._health_monitor.reset_errors("UbTtsProvider")
                self._health_monitor.heartbeat("UbTtsProvider")
                logging.info("UbTtsProvider: Recovery successful")
                return True
            logging.warning(f"UbTtsProvider: Service returned {response.status_code}")
            return False
        except Exception as e:
            logging.error(f"UbTtsProvider: Recovery failed: {e}")
            return False

    def speak(self, tts: str, interrupt: bool = True, timestamp: int = 0) -> bool:
        """Sends a request to the TTS service. Returns True on success."""
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
            success = res.get("code") == 0
            if success:
                self._health_monitor.heartbeat("UbTtsProvider")
            return success
        except requests.exceptions.RequestException as e:
            self._health_monitor.report_error("UbTtsProvider", str(e))
            logging.error(f"Failed to send TTS command: {e}")
            return False

    def get_tts_status(self, timestamp: int) -> str:
        """
        Gets the status of a specific TTS task.
        Possible statuses: 'build', 'wait', 'run', 'idle'.
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
