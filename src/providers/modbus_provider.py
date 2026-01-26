import logging
from dataclasses import dataclass
from typing import Dict, List

from pymodbus.client import AsyncModbusTcpClient

from .singleton import singleton


@dataclass
class ModbusRegisterResult:
    """
    Result of a Modbus register read operation.

    Parameters
    ----------
    address : int
        The register address that was read.
    values : List[int]
        Raw register values returned by the device.
    is_error : bool
        Whether the read operation resulted in an error.
    error_message : str
        Error description if is_error is True.
    """

    address: int
    values: List[int]
    is_error: bool = False
    error_message: str = ""


@dataclass
class ModbusCoilResult:
    """
    Result of a Modbus coil read operation.

    Parameters
    ----------
    address : int
        The coil address that was read.
    values : List[bool]
        Coil states returned by the device.
    is_error : bool
        Whether the read operation resulted in an error.
    error_message : str
        Error description if is_error is True.
    """

    address: int
    values: List[bool]
    is_error: bool = False
    error_message: str = ""


@singleton
class ModbusProvider:
    """
    Singleton provider for managing Modbus TCP client connections.

    This provider manages a pool of AsyncModbusTcpClient instances, one per
    unique host:port combination. It provides async methods for reading and
    writing Modbus registers and coils.

    pymodbus 3.x handles reconnection automatically with configurable
    exponential backoff (reconnect_delay, reconnect_delay_max).
    """

    def __init__(self) -> None:
        """Initialize the Modbus provider with an empty client pool."""
        logging.info("ModbusProvider initializing")
        self._clients: Dict[str, AsyncModbusTcpClient] = {}

    def _client_key(self, host: str, port: int) -> str:
        """
        Generate a unique key for a host:port combination.

        Parameters
        ----------
        host : str
            The Modbus device hostname or IP address.
        port : int
            The Modbus TCP port number.

        Returns
        -------
        str
            A unique key string in the format "host:port".
        """
        return f"{host}:{port}"

    async def get_client(
        self,
        host: str,
        port: int = 502,
        timeout: float = 3.0,
        retries: int = 3,
        reconnect_delay: float = 0.1,
        reconnect_delay_max: float = 300.0,
    ) -> AsyncModbusTcpClient:
        """
        Get or create an AsyncModbusTcpClient for the given host:port.

        If a client already exists for the given host:port, returns the existing
        client. Otherwise creates a new one and attempts to connect.

        Parameters
        ----------
        host : str
            The Modbus device hostname or IP address.
        port : int
            The Modbus TCP port number.
        timeout : float
            Connection and read timeout in seconds.
        retries : int
            Number of retries for failed operations.
        reconnect_delay : float
            Initial reconnection delay in seconds.
        reconnect_delay_max : float
            Maximum reconnection delay in seconds.

        Returns
        -------
        AsyncModbusTcpClient
            The Modbus TCP client instance.
        """
        key = self._client_key(host, port)

        if key not in self._clients:
            logging.info(f"Creating Modbus TCP client for {key}")
            client = AsyncModbusTcpClient(
                host,
                port=port,
                timeout=timeout,
                retries=retries,
                reconnect_delay=reconnect_delay,
                reconnect_delay_max=reconnect_delay_max,
            )
            self._clients[key] = client

        client = self._clients[key]

        if not client.connected:
            connected = await client.connect()
            if connected:
                logging.info(f"Connected to Modbus device at {key}")
            else:
                logging.warning(f"Failed to connect to Modbus device at {key}")

        return client

    async def read_holding_registers(
        self,
        host: str,
        port: int,
        address: int,
        count: int = 1,
        device_id: int = 1,
    ) -> ModbusRegisterResult:
        """
        Read holding registers from a Modbus device.

        Parameters
        ----------
        host : str
            The Modbus device hostname or IP address.
        port : int
            The Modbus TCP port number.
        address : int
            Starting register address.
        count : int
            Number of registers to read.
        device_id : int
            Modbus device (unit) ID.

        Returns
        -------
        ModbusRegisterResult
            Result containing register values or error information.
        """
        try:
            client = await self.get_client(host, port)
            if not client.connected:
                return ModbusRegisterResult(
                    address=address,
                    values=[],
                    is_error=True,
                    error_message=f"Not connected to {host}:{port}",
                )

            response = await client.read_holding_registers(
                address, count=count, device_id=device_id
            )

            if response.isError():
                return ModbusRegisterResult(
                    address=address,
                    values=[],
                    is_error=True,
                    error_message=str(response),
                )

            return ModbusRegisterResult(
                address=address,
                values=list(response.registers),
            )

        except Exception as e:
            logging.error(
                f"Error reading holding registers at {address} from {host}:{port}: {e}"
            )
            return ModbusRegisterResult(
                address=address,
                values=[],
                is_error=True,
                error_message=str(e),
            )

    async def read_input_registers(
        self,
        host: str,
        port: int,
        address: int,
        count: int = 1,
        device_id: int = 1,
    ) -> ModbusRegisterResult:
        """
        Read input registers from a Modbus device.

        Parameters
        ----------
        host : str
            The Modbus device hostname or IP address.
        port : int
            The Modbus TCP port number.
        address : int
            Starting register address.
        count : int
            Number of registers to read.
        device_id : int
            Modbus device (unit) ID.

        Returns
        -------
        ModbusRegisterResult
            Result containing register values or error information.
        """
        try:
            client = await self.get_client(host, port)
            if not client.connected:
                return ModbusRegisterResult(
                    address=address,
                    values=[],
                    is_error=True,
                    error_message=f"Not connected to {host}:{port}",
                )

            response = await client.read_input_registers(
                address, count=count, device_id=device_id
            )

            if response.isError():
                return ModbusRegisterResult(
                    address=address,
                    values=[],
                    is_error=True,
                    error_message=str(response),
                )

            return ModbusRegisterResult(
                address=address,
                values=list(response.registers),
            )

        except Exception as e:
            logging.error(
                f"Error reading input registers at {address} from {host}:{port}: {e}"
            )
            return ModbusRegisterResult(
                address=address,
                values=[],
                is_error=True,
                error_message=str(e),
            )

    async def read_coils(
        self,
        host: str,
        port: int,
        address: int,
        count: int = 1,
        device_id: int = 1,
    ) -> ModbusCoilResult:
        """
        Read coils from a Modbus device.

        Parameters
        ----------
        host : str
            The Modbus device hostname or IP address.
        port : int
            The Modbus TCP port number.
        address : int
            Starting coil address.
        count : int
            Number of coils to read.
        device_id : int
            Modbus device (unit) ID.

        Returns
        -------
        ModbusCoilResult
            Result containing coil states or error information.
        """
        try:
            client = await self.get_client(host, port)
            if not client.connected:
                return ModbusCoilResult(
                    address=address,
                    values=[],
                    is_error=True,
                    error_message=f"Not connected to {host}:{port}",
                )

            response = await client.read_coils(
                address, count=count, device_id=device_id
            )

            if response.isError():
                return ModbusCoilResult(
                    address=address,
                    values=[],
                    is_error=True,
                    error_message=str(response),
                )

            return ModbusCoilResult(
                address=address,
                values=list(response.bits[:count]),
            )

        except Exception as e:
            logging.error(f"Error reading coils at {address} from {host}:{port}: {e}")
            return ModbusCoilResult(
                address=address,
                values=[],
                is_error=True,
                error_message=str(e),
            )

    async def read_discrete_inputs(
        self,
        host: str,
        port: int,
        address: int,
        count: int = 1,
        device_id: int = 1,
    ) -> ModbusCoilResult:
        """
        Read discrete inputs from a Modbus device.

        Parameters
        ----------
        host : str
            The Modbus device hostname or IP address.
        port : int
            The Modbus TCP port number.
        address : int
            Starting discrete input address.
        count : int
            Number of discrete inputs to read.
        device_id : int
            Modbus device (unit) ID.

        Returns
        -------
        ModbusCoilResult
            Result containing discrete input states or error information.
        """
        try:
            client = await self.get_client(host, port)
            if not client.connected:
                return ModbusCoilResult(
                    address=address,
                    values=[],
                    is_error=True,
                    error_message=f"Not connected to {host}:{port}",
                )

            response = await client.read_discrete_inputs(
                address, count=count, device_id=device_id
            )

            if response.isError():
                return ModbusCoilResult(
                    address=address,
                    values=[],
                    is_error=True,
                    error_message=str(response),
                )

            return ModbusCoilResult(
                address=address,
                values=list(response.bits[:count]),
            )

        except Exception as e:
            logging.error(
                f"Error reading discrete inputs at {address} from {host}:{port}: {e}"
            )
            return ModbusCoilResult(
                address=address,
                values=[],
                is_error=True,
                error_message=str(e),
            )

    async def write_register(
        self,
        host: str,
        port: int,
        address: int,
        value: int,
        device_id: int = 1,
    ) -> bool:
        """
        Write a single holding register to a Modbus device.

        Parameters
        ----------
        host : str
            The Modbus device hostname or IP address.
        port : int
            The Modbus TCP port number.
        address : int
            Register address to write.
        value : int
            Value to write (0-65535).
        device_id : int
            Modbus device (unit) ID.

        Returns
        -------
        bool
            True if the write was successful, False otherwise.
        """
        try:
            client = await self.get_client(host, port)
            if not client.connected:
                logging.error(f"Not connected to {host}:{port}")
                return False

            response = await client.write_register(address, value, device_id=device_id)

            if response.isError():
                logging.error(
                    f"Error writing register {address}={value} to {host}:{port}: "
                    f"{response}"
                )
                return False

            logging.info(f"Written register {address}={value} to {host}:{port}")
            return True

        except Exception as e:
            logging.error(
                f"Error writing register {address}={value} to {host}:{port}: {e}"
            )
            return False

    async def write_coil(
        self,
        host: str,
        port: int,
        address: int,
        value: bool,
        device_id: int = 1,
    ) -> bool:
        """
        Write a single coil to a Modbus device.

        Parameters
        ----------
        host : str
            The Modbus device hostname or IP address.
        port : int
            The Modbus TCP port number.
        address : int
            Coil address to write.
        value : bool
            Coil state to write (True=ON, False=OFF).
        device_id : int
            Modbus device (unit) ID.

        Returns
        -------
        bool
            True if the write was successful, False otherwise.
        """
        try:
            client = await self.get_client(host, port)
            if not client.connected:
                logging.error(f"Not connected to {host}:{port}")
                return False

            response = await client.write_coil(address, value, device_id=device_id)

            if response.isError():
                logging.error(
                    f"Error writing coil {address}={value} to {host}:{port}: "
                    f"{response}"
                )
                return False

            logging.info(f"Written coil {address}={value} to {host}:{port}")
            return True

        except Exception as e:
            logging.error(f"Error writing coil {address}={value} to {host}:{port}: {e}")
            return False

    async def close_all(self) -> None:
        """
        Close all Modbus client connections.

        This method should be called during application shutdown to cleanly
        close all open connections.
        """
        for key, client in self._clients.items():
            try:
                client.close()
                logging.info(f"Closed Modbus connection to {key}")
            except Exception as e:
                logging.error(f"Error closing Modbus connection to {key}: {e}")
        self._clients.clear()
