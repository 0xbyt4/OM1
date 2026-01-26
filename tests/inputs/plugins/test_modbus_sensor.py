from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from inputs.base import Message
from inputs.plugins.modbus_sensor import (
    ModbusRegisterConfig,
    ModbusSensor,
    ModbusSensorConfig,
)
from providers.modbus_provider import ModbusCoilResult, ModbusRegisterResult


@pytest.fixture
def sensor_config():
    """Create a test sensor configuration."""
    return ModbusSensorConfig(
        host="192.168.1.10",
        port=502,
        device_id=1,
        poll_interval=0.01,
        descriptor="Test PLC Sensor",
        registers=[
            ModbusRegisterConfig(
                address=100,
                name="Temperature",
                register_type="holding",
                scale=0.1,
                unit="C",
                data_type="uint16",
            ),
            ModbusRegisterConfig(
                address=101,
                name="Pressure",
                register_type="input",
                scale=0.01,
                unit="bar",
                data_type="uint16",
            ),
            ModbusRegisterConfig(
                address=200,
                name="Motor",
                register_type="coil",
                data_type="bool",
            ),
        ],
    )


def test_initialization(sensor_config):
    """Test ModbusSensor initialization."""
    with (
        patch("inputs.plugins.modbus_sensor.ModbusProvider"),
        patch("inputs.plugins.modbus_sensor.IOProvider"),
    ):
        sensor = ModbusSensor(config=sensor_config)

        assert sensor.messages == []
        assert sensor.descriptor_for_LLM == "Test PLC Sensor"


class TestConvertRegistersToValue:
    """Tests for register value conversion."""

    def test_uint16(self, sensor_config):
        """Test uint16 conversion."""
        with (
            patch("inputs.plugins.modbus_sensor.ModbusProvider"),
            patch("inputs.plugins.modbus_sensor.IOProvider"),
        ):
            sensor = ModbusSensor(config=sensor_config)
            result = sensor._convert_registers_to_value([250], "uint16")
            assert result == 250.0

    def test_int16_positive(self, sensor_config):
        """Test int16 positive value conversion."""
        with (
            patch("inputs.plugins.modbus_sensor.ModbusProvider"),
            patch("inputs.plugins.modbus_sensor.IOProvider"),
        ):
            sensor = ModbusSensor(config=sensor_config)
            result = sensor._convert_registers_to_value([100], "int16")
            assert result == 100.0

    def test_int16_negative(self, sensor_config):
        """Test int16 negative value conversion."""
        with (
            patch("inputs.plugins.modbus_sensor.ModbusProvider"),
            patch("inputs.plugins.modbus_sensor.IOProvider"),
        ):
            sensor = ModbusSensor(config=sensor_config)
            result = sensor._convert_registers_to_value([65535], "int16")
            assert result == -1.0

    def test_uint32(self, sensor_config):
        """Test uint32 conversion from two registers."""
        with (
            patch("inputs.plugins.modbus_sensor.ModbusProvider"),
            patch("inputs.plugins.modbus_sensor.IOProvider"),
        ):
            sensor = ModbusSensor(config=sensor_config)
            result = sensor._convert_registers_to_value([1, 0], "uint32")
            assert result == 65536.0

    def test_int32_negative(self, sensor_config):
        """Test int32 negative value conversion."""
        with (
            patch("inputs.plugins.modbus_sensor.ModbusProvider"),
            patch("inputs.plugins.modbus_sensor.IOProvider"),
        ):
            sensor = ModbusSensor(config=sensor_config)
            result = sensor._convert_registers_to_value([65535, 65535], "int32")
            assert result == -1.0

    def test_float32(self, sensor_config):
        """Test float32 conversion from two registers."""
        with (
            patch("inputs.plugins.modbus_sensor.ModbusProvider"),
            patch("inputs.plugins.modbus_sensor.IOProvider"),
        ):
            sensor = ModbusSensor(config=sensor_config)
            # IEEE 754: 0x41C80000 = 25.0
            result = sensor._convert_registers_to_value([0x41C8, 0x0000], "float32")
            assert result == 25.0

    def test_empty_values(self, sensor_config):
        """Test conversion with empty values."""
        with (
            patch("inputs.plugins.modbus_sensor.ModbusProvider"),
            patch("inputs.plugins.modbus_sensor.IOProvider"),
        ):
            sensor = ModbusSensor(config=sensor_config)
            result = sensor._convert_registers_to_value([], "uint16")
            assert result == 0.0


class TestFormatValue:
    """Tests for value formatting."""

    def test_format_with_scale_and_unit(self, sensor_config):
        """Test formatting with scale and unit."""
        with (
            patch("inputs.plugins.modbus_sensor.ModbusProvider"),
            patch("inputs.plugins.modbus_sensor.IOProvider"),
        ):
            sensor = ModbusSensor(config=sensor_config)
            reg = ModbusRegisterConfig(address=100, name="Temp", scale=0.1, unit="C")
            result = sensor._format_value(250.0, reg)
            assert result == "25 C"

    def test_format_integer_value(self, sensor_config):
        """Test formatting integer values without decimals."""
        with (
            patch("inputs.plugins.modbus_sensor.ModbusProvider"),
            patch("inputs.plugins.modbus_sensor.IOProvider"),
        ):
            sensor = ModbusSensor(config=sensor_config)
            reg = ModbusRegisterConfig(address=100, name="Count", scale=1.0, unit="pcs")
            result = sensor._format_value(42.0, reg)
            assert result == "42 pcs"

    def test_format_bool_on(self, sensor_config):
        """Test formatting boolean ON value."""
        with (
            patch("inputs.plugins.modbus_sensor.ModbusProvider"),
            patch("inputs.plugins.modbus_sensor.IOProvider"),
        ):
            sensor = ModbusSensor(config=sensor_config)
            reg = ModbusRegisterConfig(address=200, name="Motor", data_type="bool")
            result = sensor._format_value(1.0, reg)
            assert result == "ON"

    def test_format_bool_off(self, sensor_config):
        """Test formatting boolean OFF value."""
        with (
            patch("inputs.plugins.modbus_sensor.ModbusProvider"),
            patch("inputs.plugins.modbus_sensor.IOProvider"),
        ):
            sensor = ModbusSensor(config=sensor_config)
            reg = ModbusRegisterConfig(address=200, name="Motor", data_type="bool")
            result = sensor._format_value(0.0, reg)
            assert result == "OFF"

    def test_format_with_offset(self, sensor_config):
        """Test formatting with scale and offset."""
        with (
            patch("inputs.plugins.modbus_sensor.ModbusProvider"),
            patch("inputs.plugins.modbus_sensor.IOProvider"),
        ):
            sensor = ModbusSensor(config=sensor_config)
            reg = ModbusRegisterConfig(
                address=100, name="Temp", scale=0.1, offset=-40.0, unit="C"
            )
            result = sensor._format_value(650.0, reg)
            assert result == "25 C"


class TestPoll:
    """Tests for ModbusSensor._poll."""

    @pytest.mark.asyncio
    async def test_poll_reads_all_registers(self, sensor_config):
        """Test that poll reads all configured registers."""
        mock_provider = AsyncMock()
        mock_provider.read_holding_registers = AsyncMock(
            return_value=ModbusRegisterResult(address=100, values=[250])
        )
        mock_provider.read_input_registers = AsyncMock(
            return_value=ModbusRegisterResult(address=101, values=[321])
        )
        mock_provider.read_coils = AsyncMock(
            return_value=ModbusCoilResult(address=200, values=[True])
        )

        with (
            patch(
                "inputs.plugins.modbus_sensor.ModbusProvider",
                return_value=mock_provider,
            ),
            patch("inputs.plugins.modbus_sensor.IOProvider"),
            patch("inputs.plugins.modbus_sensor.asyncio.sleep", new=AsyncMock()),
        ):
            sensor = ModbusSensor(config=sensor_config)
            result = await sensor._poll()

        assert result is not None
        assert "Temperature" in result
        assert "Pressure" in result
        assert "Motor" in result

    @pytest.mark.asyncio
    async def test_poll_returns_none_when_no_registers(self):
        """Test that poll returns None when no registers configured."""
        config = ModbusSensorConfig(
            host="192.168.1.10",
            port=502,
            poll_interval=0.01,
            registers=[],
        )

        with (
            patch("inputs.plugins.modbus_sensor.ModbusProvider"),
            patch("inputs.plugins.modbus_sensor.IOProvider"),
            patch("inputs.plugins.modbus_sensor.asyncio.sleep", new=AsyncMock()),
        ):
            sensor = ModbusSensor(config=config)
            result = await sensor._poll()

        assert result is None

    @pytest.mark.asyncio
    async def test_poll_skips_failed_registers(self, sensor_config):
        """Test that poll skips registers that fail to read."""
        mock_provider = AsyncMock()
        mock_provider.read_holding_registers = AsyncMock(
            return_value=ModbusRegisterResult(
                address=100, values=[], is_error=True, error_message="Timeout"
            )
        )
        mock_provider.read_input_registers = AsyncMock(
            return_value=ModbusRegisterResult(address=101, values=[321])
        )
        mock_provider.read_coils = AsyncMock(
            return_value=ModbusCoilResult(address=200, values=[True])
        )

        with (
            patch(
                "inputs.plugins.modbus_sensor.ModbusProvider",
                return_value=mock_provider,
            ),
            patch("inputs.plugins.modbus_sensor.IOProvider"),
            patch("inputs.plugins.modbus_sensor.asyncio.sleep", new=AsyncMock()),
        ):
            sensor = ModbusSensor(config=sensor_config)
            result = await sensor._poll()

        assert result is not None
        assert "Temperature" not in result
        assert "Pressure" in result
        assert "Motor" in result


class TestRawToText:
    """Tests for ModbusSensor._raw_to_text and raw_to_text."""

    @pytest.mark.asyncio
    async def test_raw_to_text_with_data(self, sensor_config):
        """Test _raw_to_text with valid data."""
        with (
            patch("inputs.plugins.modbus_sensor.ModbusProvider"),
            patch("inputs.plugins.modbus_sensor.IOProvider"),
            patch("inputs.plugins.modbus_sensor.time.time", return_value=1000.0),
        ):
            sensor = ModbusSensor(config=sensor_config)
            raw = {"Temperature": "25.00 C", "Pressure": "3.21 bar", "Motor": "ON"}

            result = await sensor._raw_to_text(raw)

            assert result is not None
            assert result.timestamp == 1000.0
            assert "Temperature: 25.00 C" in result.message
            assert "Pressure: 3.21 bar" in result.message
            assert "Motor: ON" in result.message

    @pytest.mark.asyncio
    async def test_raw_to_text_with_none(self, sensor_config):
        """Test _raw_to_text with None input."""
        with (
            patch("inputs.plugins.modbus_sensor.ModbusProvider"),
            patch("inputs.plugins.modbus_sensor.IOProvider"),
        ):
            sensor = ModbusSensor(config=sensor_config)
            result = await sensor._raw_to_text(None)
            assert result is None

    @pytest.mark.asyncio
    async def test_raw_to_text_updates_buffer(self, sensor_config):
        """Test that raw_to_text appends to messages buffer."""
        with (
            patch("inputs.plugins.modbus_sensor.ModbusProvider"),
            patch("inputs.plugins.modbus_sensor.IOProvider"),
            patch("inputs.plugins.modbus_sensor.time.time", return_value=1000.0),
        ):
            sensor = ModbusSensor(config=sensor_config)
            raw = {"Temperature": "25.00 C"}

            await sensor.raw_to_text(raw)

            assert len(sensor.messages) == 1
            assert "Temperature: 25.00 C" in sensor.messages[0].message


class TestFormattedLatestBuffer:
    """Tests for ModbusSensor.formatted_latest_buffer."""

    def test_formatted_latest_buffer_with_messages(self, sensor_config):
        """Test formatted_latest_buffer with messages."""
        with (
            patch("inputs.plugins.modbus_sensor.ModbusProvider"),
            patch("inputs.plugins.modbus_sensor.IOProvider"),
        ):
            sensor = ModbusSensor(config=sensor_config)
            sensor.io_provider = MagicMock()

            sensor.messages = [
                Message(
                    timestamp=1000.0,
                    message="Temperature: 25.00 C, Motor: ON",
                ),
            ]

            result = sensor.formatted_latest_buffer()

            assert result is not None
            assert "Test PLC Sensor" in result
            assert "Temperature: 25.00 C" in result
            assert "Motor: ON" in result
            sensor.io_provider.add_input.assert_called_once()
            assert len(sensor.messages) == 0

    def test_formatted_latest_buffer_empty(self, sensor_config):
        """Test formatted_latest_buffer with empty buffer."""
        with (
            patch("inputs.plugins.modbus_sensor.ModbusProvider"),
            patch("inputs.plugins.modbus_sensor.IOProvider"),
        ):
            sensor = ModbusSensor(config=sensor_config)
            result = sensor.formatted_latest_buffer()
            assert result is None
