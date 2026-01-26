from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from providers.modbus_provider import (
    ModbusProvider,
)


@pytest.fixture(autouse=True)
def reset_singleton():
    """Reset singleton instances between tests."""
    ModbusProvider.reset()  # type: ignore
    yield
    ModbusProvider.reset()  # type: ignore


@pytest.fixture
def mock_client():
    """Create a mock AsyncModbusTcpClient."""
    client = AsyncMock()
    client.connected = True
    client.connect = AsyncMock(return_value=True)
    client.close = MagicMock()

    # Default successful register response
    register_response = MagicMock()
    register_response.isError.return_value = False
    register_response.registers = [250, 321, 1]
    client.read_holding_registers = AsyncMock(return_value=register_response)
    client.read_input_registers = AsyncMock(return_value=register_response)

    # Default successful coil response
    coil_response = MagicMock()
    coil_response.isError.return_value = False
    coil_response.bits = [True, False, True]
    client.read_coils = AsyncMock(return_value=coil_response)
    client.read_discrete_inputs = AsyncMock(return_value=coil_response)

    # Default successful write response
    write_response = MagicMock()
    write_response.isError.return_value = False
    client.write_register = AsyncMock(return_value=write_response)
    client.write_coil = AsyncMock(return_value=write_response)

    return client


class TestModbusProviderInit:
    """Tests for ModbusProvider initialization."""

    def test_singleton_pattern(self):
        """Test that ModbusProvider follows singleton pattern."""
        provider1 = ModbusProvider()
        provider2 = ModbusProvider()
        assert provider1 is provider2

    def test_initialization(self):
        """Test ModbusProvider initializes with empty client pool."""
        provider = ModbusProvider()
        assert provider._clients == {}


class TestModbusProviderGetClient:
    """Tests for ModbusProvider.get_client."""

    @pytest.mark.asyncio
    async def test_creates_new_client(self, mock_client):
        """Test creating a new client for a host:port."""
        with patch(
            "providers.modbus_provider.AsyncModbusTcpClient",
            return_value=mock_client,
        ):
            provider = ModbusProvider()
            client = await provider.get_client("192.168.1.10", 502)

            assert client is mock_client
            assert "192.168.1.10:502" in provider._clients

    @pytest.mark.asyncio
    async def test_reuses_existing_client(self, mock_client):
        """Test that same host:port returns existing client."""
        with patch(
            "providers.modbus_provider.AsyncModbusTcpClient",
            return_value=mock_client,
        ):
            provider = ModbusProvider()
            client1 = await provider.get_client("192.168.1.10", 502)
            client2 = await provider.get_client("192.168.1.10", 502)

            assert client1 is client2

    @pytest.mark.asyncio
    async def test_different_hosts_different_clients(self, mock_client):
        """Test that different host:port combinations get separate clients."""
        mock_client2 = AsyncMock()
        mock_client2.connected = True
        mock_client2.connect = AsyncMock(return_value=True)

        clients = [mock_client, mock_client2]
        call_count = 0

        def create_client(*args, **kwargs):
            nonlocal call_count
            c = clients[call_count]
            call_count += 1
            return c

        with patch(
            "providers.modbus_provider.AsyncModbusTcpClient",
            side_effect=create_client,
        ):
            provider = ModbusProvider()
            c1 = await provider.get_client("192.168.1.10", 502)
            c2 = await provider.get_client("192.168.1.20", 502)

            assert c1 is not c2
            assert len(provider._clients) == 2

    @pytest.mark.asyncio
    async def test_connects_when_disconnected(self, mock_client):
        """Test that get_client connects when client is disconnected."""
        mock_client.connected = False
        with patch(
            "providers.modbus_provider.AsyncModbusTcpClient",
            return_value=mock_client,
        ):
            provider = ModbusProvider()
            await provider.get_client("192.168.1.10", 502)

            mock_client.connect.assert_called_once()


class TestModbusProviderReadRegisters:
    """Tests for ModbusProvider register read operations."""

    @pytest.mark.asyncio
    async def test_read_holding_registers_success(self, mock_client):
        """Test successful holding register read."""
        with patch(
            "providers.modbus_provider.AsyncModbusTcpClient",
            return_value=mock_client,
        ):
            provider = ModbusProvider()
            result = await provider.read_holding_registers(
                "192.168.1.10", 502, address=100, count=3
            )

            assert not result.is_error
            assert result.values == [250, 321, 1]
            assert result.address == 100

    @pytest.mark.asyncio
    async def test_read_holding_registers_error(self, mock_client):
        """Test handling of holding register read error."""
        error_response = MagicMock()
        error_response.isError.return_value = True
        mock_client.read_holding_registers = AsyncMock(return_value=error_response)

        with patch(
            "providers.modbus_provider.AsyncModbusTcpClient",
            return_value=mock_client,
        ):
            provider = ModbusProvider()
            result = await provider.read_holding_registers(
                "192.168.1.10", 502, address=100
            )

            assert result.is_error
            assert result.values == []

    @pytest.mark.asyncio
    async def test_read_holding_registers_disconnected(self, mock_client):
        """Test read when client is disconnected."""
        mock_client.connected = False
        mock_client.connect = AsyncMock(return_value=False)

        with patch(
            "providers.modbus_provider.AsyncModbusTcpClient",
            return_value=mock_client,
        ):
            provider = ModbusProvider()
            result = await provider.read_holding_registers(
                "192.168.1.10", 502, address=100
            )

            assert result.is_error
            assert "Not connected" in result.error_message

    @pytest.mark.asyncio
    async def test_read_input_registers_success(self, mock_client):
        """Test successful input register read."""
        with patch(
            "providers.modbus_provider.AsyncModbusTcpClient",
            return_value=mock_client,
        ):
            provider = ModbusProvider()
            result = await provider.read_input_registers(
                "192.168.1.10", 502, address=200, count=2
            )

            assert not result.is_error
            assert result.values == [250, 321, 1]

    @pytest.mark.asyncio
    async def test_read_coils_success(self, mock_client):
        """Test successful coil read."""
        with patch(
            "providers.modbus_provider.AsyncModbusTcpClient",
            return_value=mock_client,
        ):
            provider = ModbusProvider()
            result = await provider.read_coils("192.168.1.10", 502, address=0, count=2)

            assert not result.is_error
            assert result.values == [True, False]

    @pytest.mark.asyncio
    async def test_read_discrete_inputs_success(self, mock_client):
        """Test successful discrete input read."""
        with patch(
            "providers.modbus_provider.AsyncModbusTcpClient",
            return_value=mock_client,
        ):
            provider = ModbusProvider()
            result = await provider.read_discrete_inputs(
                "192.168.1.10", 502, address=10, count=3
            )

            assert not result.is_error
            assert result.values == [True, False, True]

    @pytest.mark.asyncio
    async def test_read_holding_registers_exception(self, mock_client):
        """Test exception handling during register read."""
        mock_client.read_holding_registers = AsyncMock(
            side_effect=Exception("Connection lost")
        )

        with patch(
            "providers.modbus_provider.AsyncModbusTcpClient",
            return_value=mock_client,
        ):
            provider = ModbusProvider()
            result = await provider.read_holding_registers(
                "192.168.1.10", 502, address=100
            )

            assert result.is_error
            assert "Connection lost" in result.error_message


class TestModbusProviderWriteOperations:
    """Tests for ModbusProvider write operations."""

    @pytest.mark.asyncio
    async def test_write_register_success(self, mock_client):
        """Test successful register write."""
        with patch(
            "providers.modbus_provider.AsyncModbusTcpClient",
            return_value=mock_client,
        ):
            provider = ModbusProvider()
            success = await provider.write_register(
                "192.168.1.10", 502, address=100, value=1500
            )

            assert success
            mock_client.write_register.assert_called_once_with(100, 1500, device_id=1)

    @pytest.mark.asyncio
    async def test_write_register_error(self, mock_client):
        """Test handling of register write error."""
        error_response = MagicMock()
        error_response.isError.return_value = True
        mock_client.write_register = AsyncMock(return_value=error_response)

        with patch(
            "providers.modbus_provider.AsyncModbusTcpClient",
            return_value=mock_client,
        ):
            provider = ModbusProvider()
            success = await provider.write_register(
                "192.168.1.10", 502, address=100, value=1500
            )

            assert not success

    @pytest.mark.asyncio
    async def test_write_coil_success(self, mock_client):
        """Test successful coil write."""
        with patch(
            "providers.modbus_provider.AsyncModbusTcpClient",
            return_value=mock_client,
        ):
            provider = ModbusProvider()
            success = await provider.write_coil(
                "192.168.1.10", 502, address=200, value=True
            )

            assert success
            mock_client.write_coil.assert_called_once_with(200, True, device_id=1)

    @pytest.mark.asyncio
    async def test_write_register_disconnected(self, mock_client):
        """Test write when client is disconnected."""
        mock_client.connected = False
        mock_client.connect = AsyncMock(return_value=False)

        with patch(
            "providers.modbus_provider.AsyncModbusTcpClient",
            return_value=mock_client,
        ):
            provider = ModbusProvider()
            success = await provider.write_register(
                "192.168.1.10", 502, address=100, value=1500
            )

            assert not success


class TestModbusProviderCloseAll:
    """Tests for ModbusProvider.close_all."""

    @pytest.mark.asyncio
    async def test_close_all_clients(self, mock_client):
        """Test closing all client connections."""
        with patch(
            "providers.modbus_provider.AsyncModbusTcpClient",
            return_value=mock_client,
        ):
            provider = ModbusProvider()
            await provider.get_client("192.168.1.10", 502)

            await provider.close_all()

            mock_client.close.assert_called_once()
            assert provider._clients == {}
