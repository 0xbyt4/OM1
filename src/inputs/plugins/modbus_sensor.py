import asyncio
import logging
import struct
import time
from typing import Dict, List, Optional

from pydantic import Field

from inputs.base import Message, SensorConfig
from inputs.base.loop import FuserInput
from providers.io_provider import IOProvider
from providers.modbus_provider import ModbusProvider


class ModbusRegisterConfig(SensorConfig):
    """
    Configuration for a single Modbus register.

    Parameters
    ----------
    address : int
        The Modbus register address.
    name : str
        Human-readable name for this register.
    register_type : str
        Type of register: "holding", "input", "coil", or "discrete".
    count : int
        Number of registers to read (default 1).
    scale : float
        Multiplier applied to raw value (default 1.0).
    offset : float
        Additive offset applied after scaling (default 0.0).
    unit : str
        Unit of measurement for display (default "").
    data_type : str
        Data type interpretation: "uint16", "int16", "uint32", "int32",
        "float32", or "bool" (default "uint16").
    """

    address: int
    name: str
    register_type: str = Field(default="holding")
    count: int = Field(default=1)
    scale: float = Field(default=1.0)
    offset: float = Field(default=0.0)
    unit: str = Field(default="")
    data_type: str = Field(default="uint16")


class ModbusSensorConfig(SensorConfig):
    """
    Configuration for the Modbus sensor input plugin.

    Parameters
    ----------
    host : str
        The Modbus device hostname or IP address.
    port : int
        The Modbus TCP port number (default 502).
    device_id : int
        Modbus device (unit) ID (default 1).
    poll_interval : float
        Seconds between register polls (default 1.0).
    registers : List[ModbusRegisterConfig]
        List of register configurations to read.
    descriptor : str
        Human-readable name for LLM context (default "Modbus PLC Sensor").
    """

    host: str
    port: int = Field(default=502)
    device_id: int = Field(default=1)
    poll_interval: float = Field(default=1.0)
    registers: List[ModbusRegisterConfig] = Field(default_factory=list)
    descriptor: str = Field(default="Modbus PLC Sensor")


class ModbusSensor(FuserInput[ModbusSensorConfig, Optional[Dict[str, str]]]):
    """
    Modbus TCP input sensor for reading PLC registers.

    This plugin reads Modbus registers (holding, input, coils, discrete inputs)
    from a PLC or industrial device and formats the data as human-readable text
    for LLM processing.

    Supports register types: holding registers, input registers, coils, and
    discrete inputs with configurable scaling, offset, and data type conversion.
    """

    def __init__(self, config: ModbusSensorConfig):
        """
        Initialize the Modbus sensor.

        Parameters
        ----------
        config : ModbusSensorConfig
            Configuration containing host, port, registers, and poll settings.
        """
        super().__init__(config)

        self.modbus_provider = ModbusProvider()
        self.io_provider = IOProvider()
        self.messages: list[Message] = []
        self.descriptor_for_LLM = config.descriptor

    def _convert_registers_to_value(
        self,
        raw_values: List[int],
        data_type: str,
    ) -> float:
        """
        Convert raw register values to a typed numeric value.

        Parameters
        ----------
        raw_values : List[int]
            Raw 16-bit register values from the device.
        data_type : str
            Target data type for interpretation.

        Returns
        -------
        float
            The converted numeric value.
        """
        if not raw_values:
            return 0.0

        if data_type == "int16":
            raw = raw_values[0]
            if raw >= 32768:
                raw -= 65536
            return float(raw)

        if data_type == "uint16":
            return float(raw_values[0])

        if data_type == "uint32" and len(raw_values) >= 2:
            return float((raw_values[0] << 16) | raw_values[1])

        if data_type == "int32" and len(raw_values) >= 2:
            combined = (raw_values[0] << 16) | raw_values[1]
            if combined >= 2147483648:
                combined -= 4294967296
            return float(combined)

        if data_type == "float32" and len(raw_values) >= 2:
            packed = struct.pack(">HH", raw_values[0], raw_values[1])
            return float(struct.unpack(">f", packed)[0])

        return float(raw_values[0])

    def _format_value(
        self,
        value: float,
        register_config: ModbusRegisterConfig,
    ) -> str:
        """
        Format a register value with scaling, offset, and unit.

        Parameters
        ----------
        value : float
            Raw numeric value from the register.
        register_config : ModbusRegisterConfig
            Register configuration for formatting.

        Returns
        -------
        str
            Formatted string like "25.0°C" or "ON".
        """
        scaled = (value * register_config.scale) + register_config.offset

        if register_config.data_type == "bool":
            state = "ON" if scaled != 0 else "OFF"
            return state

        if scaled == int(scaled):
            formatted = str(int(scaled))
        else:
            formatted = f"{scaled:.2f}"

        if register_config.unit:
            return f"{formatted} {register_config.unit}"
        return formatted

    async def _read_register(self, reg: ModbusRegisterConfig) -> Optional[str]:
        """
        Read a single register configuration from the Modbus device.

        Parameters
        ----------
        reg : ModbusRegisterConfig
            The register configuration to read.

        Returns
        -------
        Optional[str]
            Formatted value string, or None if the read failed.
        """
        host = self.config.host
        port = self.config.port
        device_id = self.config.device_id

        if reg.register_type == "holding":
            result = await self.modbus_provider.read_holding_registers(
                host, port, reg.address, count=reg.count, device_id=device_id
            )
            if result.is_error:
                logging.warning(
                    f"Modbus read error for {reg.name}: {result.error_message}"
                )
                return None
            value = self._convert_registers_to_value(result.values, reg.data_type)

        elif reg.register_type == "input":
            result = await self.modbus_provider.read_input_registers(
                host, port, reg.address, count=reg.count, device_id=device_id
            )
            if result.is_error:
                logging.warning(
                    f"Modbus read error for {reg.name}: {result.error_message}"
                )
                return None
            value = self._convert_registers_to_value(result.values, reg.data_type)

        elif reg.register_type == "coil":
            coil_result = await self.modbus_provider.read_coils(
                host, port, reg.address, count=reg.count, device_id=device_id
            )
            if coil_result.is_error:
                logging.warning(
                    f"Modbus read error for {reg.name}: {coil_result.error_message}"
                )
                return None
            value = 1.0 if coil_result.values and coil_result.values[0] else 0.0

        elif reg.register_type == "discrete":
            discrete_result = await self.modbus_provider.read_discrete_inputs(
                host, port, reg.address, count=reg.count, device_id=device_id
            )
            if discrete_result.is_error:
                logging.warning(
                    f"Modbus read error for {reg.name}: "
                    f"{discrete_result.error_message}"
                )
                return None
            value = 1.0 if discrete_result.values and discrete_result.values[0] else 0.0

        else:
            logging.warning(f"Unknown register type: {reg.register_type}")
            return None

        return self._format_value(value, reg)

    async def _poll(self) -> Optional[Dict[str, str]]:
        """
        Poll all configured Modbus registers.

        Returns
        -------
        Optional[Dict[str, str]]
            Dictionary mapping register names to formatted values,
            or None if no registers could be read.
        """
        await asyncio.sleep(self.config.poll_interval)

        if not self.config.registers:
            return None

        readings: Dict[str, str] = {}

        for reg in self.config.registers:
            formatted_value = await self._read_register(reg)
            if formatted_value is not None:
                readings[reg.name] = formatted_value

        if not readings:
            return None

        return readings

    async def _raw_to_text(
        self, raw_input: Optional[Dict[str, str]]
    ) -> Optional[Message]:
        """
        Convert raw register readings to a human-readable message.

        Parameters
        ----------
        raw_input : Optional[Dict[str, str]]
            Dictionary of register name to formatted value pairs.

        Returns
        -------
        Optional[Message]
            Timestamped message with formatted register data.
        """
        if raw_input is None:
            return None

        parts = [f"{name}: {value}" for name, value in raw_input.items()]
        message_text = ", ".join(parts)

        return Message(timestamp=time.time(), message=message_text)

    async def raw_to_text(self, raw_input: Optional[Dict[str, str]]):
        """
        Process raw input and update the message buffer.

        Parameters
        ----------
        raw_input : Optional[Dict[str, str]]
            Raw register readings to process.
        """
        pending_message = await self._raw_to_text(raw_input)

        if pending_message is not None:
            self.messages.append(pending_message)

    def formatted_latest_buffer(self) -> Optional[str]:
        """
        Format the latest buffer contents for LLM consumption and clear the buffer.

        Returns
        -------
        Optional[str]
            Formatted string with register data, or None if buffer is empty.
        """
        if len(self.messages) == 0:
            return None

        latest_message = self.messages[-1]

        result = f"""
INPUT: {self.descriptor_for_LLM}
// START
{latest_message.message}
// END
"""

        self.io_provider.add_input(
            self.__class__.__name__, latest_message.message, latest_message.timestamp
        )
        self.messages = []

        return result
