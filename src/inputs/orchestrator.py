import asyncio
import logging
from collections.abc import Sequence

from inputs.base import Sensor
from providers.health_monitor_provider import HealthMonitorProvider


class InputOrchestrator:
    """
    Manages and coordinates multiple input sources.

    Handles concurrent processing of multiple Sensor instances,
    orchestrating their data flows.

    Parameters
    ----------
    inputs : Sequence[Sensor]
        Sequence of input sources to manage
    """

    inputs: Sequence[Sensor]

    def __init__(self, inputs: Sequence[Sensor]):
        """
        Initialize InputOrchestrator instance with input sources.
        """
        self.inputs = inputs
        self._health_monitor = HealthMonitorProvider()

        for input_sensor in self.inputs:
            input_name = type(input_sensor).__name__
            self._health_monitor.register(input_name, metadata={"type": "input_sensor"})

    async def listen(self) -> None:
        """
        Start listening to all input sources concurrently.

        Creates and manages async tasks for each input source.
        """
        input_tasks = [
            asyncio.create_task(self._listen_to_input(input)) for input in self.inputs
        ]
        await asyncio.gather(*input_tasks)

    async def _listen_to_input(self, input: Sensor) -> None:
        """
        Process events from a single input source.

        Parameters
        ----------
        input : Sensor
            Input source to listen to
        """
        input_name = type(input).__name__
        try:
            async for event in input.listen():
                await input.raw_to_text(event)
                self._health_monitor.heartbeat(input_name)
        except Exception as e:
            self._health_monitor.report_error(input_name, str(e))
            logging.error(f"Input {input_name} failed: {e}")
