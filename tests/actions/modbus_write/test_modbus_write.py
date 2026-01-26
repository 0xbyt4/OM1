from unittest.mock import AsyncMock, patch

import pytest

from actions.modbus_write.connector.modbus import (
    ModbusWriteConfig,
    ModbusWriteConnector,
)
from actions.modbus_write.interface import ModbusWrite, ModbusWriteInput


class TestModbusWriteInterface:
    """Tests for ModbusWrite interface."""

    def test_interface_creation(self):
        """Test creating ModbusWriteInput."""
        inp = ModbusWriteInput(action="register:100:1500")
        assert inp.action == "register:100:1500"

    def test_interface_dataclass(self):
        """Test ModbusWrite interface is a valid dataclass."""
        inp = ModbusWriteInput(action="coil:200:true")
        iface = ModbusWrite(input=inp, output=inp)
        assert iface.input.action == "coil:200:true"


class TestModbusWriteConnectorParseAction:
    """Tests for action string parsing."""

    @pytest.fixture
    def connector(self):
        """Create a ModbusWriteConnector with mocked provider."""
        with patch(
            "actions.modbus_write.connector.modbus.ModbusProvider",
        ):
            config = ModbusWriteConfig(host="192.168.1.10", port=502, device_id=1)
            return ModbusWriteConnector(config)

    def test_parse_register_write(self, connector):
        """Test parsing register write action."""
        result = connector._parse_action("register:100:1500")
        assert result is not None
        reg_type, address, value = result
        assert reg_type == "register"
        assert address == 100
        assert value == 1500

    def test_parse_coil_true(self, connector):
        """Test parsing coil write with true value."""
        result = connector._parse_action("coil:200:true")
        assert result is not None
        reg_type, address, value = result
        assert reg_type == "coil"
        assert address == 200
        assert value is True

    def test_parse_coil_false(self, connector):
        """Test parsing coil write with false value."""
        result = connector._parse_action("coil:200:false")
        assert result is not None
        _, _, value = result
        assert value is False

    def test_parse_coil_on_off(self, connector):
        """Test parsing coil write with on/off values."""
        result_on = connector._parse_action("coil:200:on")
        assert result_on is not None
        assert result_on[2] is True

        result_off = connector._parse_action("coil:200:off")
        assert result_off is not None
        assert result_off[2] is False

    def test_parse_coil_numeric(self, connector):
        """Test parsing coil write with 1/0 values."""
        result_1 = connector._parse_action("coil:200:1")
        assert result_1 is not None
        assert result_1[2] is True

        result_0 = connector._parse_action("coil:200:0")
        assert result_0 is not None
        assert result_0[2] is False

    def test_parse_invalid_format_too_few_parts(self, connector):
        """Test parsing with too few parts."""
        result = connector._parse_action("register:100")
        assert result is None

    def test_parse_invalid_format_too_many_parts(self, connector):
        """Test parsing with too many parts."""
        result = connector._parse_action("register:100:1500:extra")
        assert result is None

    def test_parse_invalid_address(self, connector):
        """Test parsing with invalid address."""
        result = connector._parse_action("register:abc:1500")
        assert result is None

    def test_parse_invalid_register_value(self, connector):
        """Test parsing with invalid register value."""
        result = connector._parse_action("register:100:notanumber")
        assert result is None

    def test_parse_register_value_out_of_range(self, connector):
        """Test parsing with register value out of uint16 range."""
        result = connector._parse_action("register:100:70000")
        assert result is None

    def test_parse_unknown_register_type(self, connector):
        """Test parsing with unknown register type."""
        result = connector._parse_action("unknown:100:1500")
        assert result is None

    def test_parse_invalid_coil_value(self, connector):
        """Test parsing with invalid coil value."""
        result = connector._parse_action("coil:200:maybe")
        assert result is None

    def test_parse_float_register_value(self, connector):
        """Test parsing register value with float truncation."""
        result = connector._parse_action("register:100:1500.7")
        assert result is not None
        assert result[2] == 1500

    def test_parse_with_whitespace(self, connector):
        """Test parsing with extra whitespace."""
        result = connector._parse_action(" register : 100 : 1500 ")
        assert result is not None
        assert result == ("register", 100, 1500)


class TestModbusWriteConnectorConnect:
    """Tests for ModbusWriteConnector.connect."""

    @pytest.mark.asyncio
    async def test_write_register_success(self):
        """Test successful register write via connect."""
        mock_provider = AsyncMock()
        mock_provider.write_register = AsyncMock(return_value=True)

        with patch(
            "actions.modbus_write.connector.modbus.ModbusProvider",
            return_value=mock_provider,
        ):
            config = ModbusWriteConfig(host="192.168.1.10", port=502, device_id=1)
            connector = ModbusWriteConnector(config)
            inp = ModbusWriteInput(action="register:100:1500")

            await connector.connect(inp)

            mock_provider.write_register.assert_called_once_with(
                "192.168.1.10", 502, 100, 1500, device_id=1
            )

    @pytest.mark.asyncio
    async def test_write_coil_success(self):
        """Test successful coil write via connect."""
        mock_provider = AsyncMock()
        mock_provider.write_coil = AsyncMock(return_value=True)

        with patch(
            "actions.modbus_write.connector.modbus.ModbusProvider",
            return_value=mock_provider,
        ):
            config = ModbusWriteConfig(host="192.168.1.10", port=502, device_id=1)
            connector = ModbusWriteConnector(config)
            inp = ModbusWriteInput(action="coil:200:true")

            await connector.connect(inp)

            mock_provider.write_coil.assert_called_once_with(
                "192.168.1.10", 502, 200, True, device_id=1
            )

    @pytest.mark.asyncio
    async def test_write_invalid_action_no_crash(self):
        """Test that invalid action string does not crash."""
        mock_provider = AsyncMock()

        with patch(
            "actions.modbus_write.connector.modbus.ModbusProvider",
            return_value=mock_provider,
        ):
            config = ModbusWriteConfig(host="192.168.1.10", port=502, device_id=1)
            connector = ModbusWriteConnector(config)
            inp = ModbusWriteInput(action="invalid")

            await connector.connect(inp)

            mock_provider.write_register.assert_not_called()
            mock_provider.write_coil.assert_not_called()

    @pytest.mark.asyncio
    async def test_write_register_failure(self):
        """Test handling of register write failure."""
        mock_provider = AsyncMock()
        mock_provider.write_register = AsyncMock(return_value=False)

        with patch(
            "actions.modbus_write.connector.modbus.ModbusProvider",
            return_value=mock_provider,
        ):
            config = ModbusWriteConfig(host="192.168.1.10", port=502, device_id=1)
            connector = ModbusWriteConnector(config)
            inp = ModbusWriteInput(action="register:100:1500")

            await connector.connect(inp)

            mock_provider.write_register.assert_called_once()

    @pytest.mark.asyncio
    async def test_write_uses_config_device_id(self):
        """Test that connector uses device_id from config."""
        mock_provider = AsyncMock()
        mock_provider.write_register = AsyncMock(return_value=True)

        with patch(
            "actions.modbus_write.connector.modbus.ModbusProvider",
            return_value=mock_provider,
        ):
            config = ModbusWriteConfig(host="10.0.0.1", port=5020, device_id=5)
            connector = ModbusWriteConnector(config)
            inp = ModbusWriteInput(action="register:50:100")

            await connector.connect(inp)

            mock_provider.write_register.assert_called_once_with(
                "10.0.0.1", 5020, 50, 100, device_id=5
            )
