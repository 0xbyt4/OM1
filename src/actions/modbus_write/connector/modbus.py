import logging
from typing import Optional, Tuple, Union

from actions.base import ActionConfig, ActionConnector
from actions.modbus_write.interface import ModbusWriteInput
from providers.modbus_provider import ModbusProvider


class ModbusWriteConfig(ActionConfig):
    """
    Configuration for the Modbus write connector.

    Parameters
    ----------
    host : str
        The Modbus device hostname or IP address.
    port : int
        The Modbus TCP port number (default 502).
    device_id : int
        Modbus device (unit) ID (default 1).
    """

    host: str = "127.0.0.1"
    port: int = 502
    device_id: int = 1


class ModbusWriteConnector(ActionConnector[ModbusWriteConfig, ModbusWriteInput]):
    """
    Connector for writing values to Modbus PLC registers and coils.

    Parses action strings in the format "register:<address>:<value>" or
    "coil:<address>:<true|false>" and writes the corresponding values
    to the Modbus device via ModbusProvider.
    """

    def __init__(self, config: ModbusWriteConfig):
        """
        Initialize the Modbus write connector.

        Parameters
        ----------
        config : ModbusWriteConfig
            Configuration containing host, port, and device_id.
        """
        super().__init__(config)
        self.modbus_provider = ModbusProvider()

    def _parse_action(self, action: str) -> Optional[Tuple[str, int, Union[int, bool]]]:
        """
        Parse the action string into register type, address, and value.

        Parameters
        ----------
        action : str
            Action string in format "register:<address>:<value>" or
            "coil:<address>:<true|false>".

        Returns
        -------
        Optional[Tuple[str, int, Union[int, bool]]]
            Tuple of (register_type, address, value) or None if parsing fails.
        """
        parts = action.strip().split(":")
        if len(parts) != 3:
            logging.error(
                f"Invalid Modbus write format: '{action}'. "
                f"Expected 'register:<address>:<value>' or "
                f"'coil:<address>:<true|false>'"
            )
            return None

        reg_type = parts[0].lower().strip()
        address_str = parts[1].strip()
        value_str = parts[2].strip()

        try:
            address = int(address_str)
        except ValueError:
            logging.error(f"Invalid register address: '{address_str}'")
            return None

        if reg_type == "register":
            try:
                value = int(value_str)
            except ValueError:
                try:
                    value = int(float(value_str))
                except ValueError:
                    logging.error(f"Invalid register value: '{value_str}'")
                    return None

            if not 0 <= value <= 65535:
                logging.error(f"Register value {value} out of range (0-65535)")
                return None

            return (reg_type, address, value)

        elif reg_type == "coil":
            value_lower = value_str.lower()
            if value_lower in ("true", "1", "on"):
                return (reg_type, address, True)
            elif value_lower in ("false", "0", "off"):
                return (reg_type, address, False)
            else:
                logging.error(
                    f"Invalid coil value: '{value_str}'. "
                    f"Expected true/false, 1/0, or on/off"
                )
                return None

        else:
            logging.error(
                f"Unknown register type: '{reg_type}'. "
                f"Expected 'register' or 'coil'"
            )
            return None

    async def connect(self, output_interface: ModbusWriteInput) -> None:
        """
        Execute the Modbus write action.

        Parses the action string from the output interface and writes the
        value to the specified register or coil on the Modbus device.

        Parameters
        ----------
        output_interface : ModbusWriteInput
            The write command containing the action string.
        """
        parsed = self._parse_action(output_interface.action)
        if parsed is None:
            return

        reg_type, address, value = parsed
        host = self.config.host
        port = self.config.port
        device_id = self.config.device_id

        if reg_type == "register":
            success = await self.modbus_provider.write_register(
                host, port, address, value, device_id=device_id
            )
            if success:
                logging.info(
                    f"Modbus write register {address}={value} "
                    f"on {host}:{port} device {device_id}"
                )
            else:
                logging.error(
                    f"Failed to write register {address}={value} "
                    f"on {host}:{port} device {device_id}"
                )

        elif reg_type == "coil":
            success = await self.modbus_provider.write_coil(
                host, port, address, bool(value), device_id=device_id
            )
            if success:
                logging.info(
                    f"Modbus write coil {address}={value} "
                    f"on {host}:{port} device {device_id}"
                )
            else:
                logging.error(
                    f"Failed to write coil {address}={value} "
                    f"on {host}:{port} device {device_id}"
                )
