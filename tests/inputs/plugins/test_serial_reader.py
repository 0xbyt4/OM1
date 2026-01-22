from unittest.mock import MagicMock, patch

import pytest

from inputs.plugins.serial_reader import (
    DEFAULT_BAUDRATE,
    DEFAULT_SERIAL_PORT,
    DEFAULT_TIMEOUT,
    SerialReader,
    SerialReaderConfig,
)


class TestSerialReaderConfig:
    """Tests for SerialReaderConfig class."""

    def test_default_values(self):
        """Test that default values are set correctly."""
        config = SerialReaderConfig()

        assert config.port == DEFAULT_SERIAL_PORT
        assert config.baudrate == DEFAULT_BAUDRATE
        assert config.timeout == DEFAULT_TIMEOUT
        assert config.descriptor == "Heart Rate and Grip Strength"

    def test_custom_values(self):
        """Test that custom values override defaults."""
        config = SerialReaderConfig(
            port="/dev/ttyACM0",
            baudrate=115200,
            timeout=2.5,
            descriptor="Custom Sensor",
        )

        assert config.port == "/dev/ttyACM0"
        assert config.baudrate == 115200
        assert config.timeout == 2.5
        assert config.descriptor == "Custom Sensor"

    def test_partial_custom_values(self):
        """Test that partial custom values work with defaults."""
        config = SerialReaderConfig(port="COM3", baudrate=19200)

        assert config.port == "COM3"
        assert config.baudrate == 19200
        assert config.timeout == DEFAULT_TIMEOUT
        assert config.descriptor == "Heart Rate and Grip Strength"


class TestSerialReader:
    """Tests for SerialReader class."""

    @patch("inputs.plugins.serial_reader.serial.Serial")
    def test_init_with_default_config(self, mock_serial):
        """Test initialization with default config."""
        config = SerialReaderConfig()

        reader = SerialReader(config)

        mock_serial.assert_called_once_with(
            DEFAULT_SERIAL_PORT, DEFAULT_BAUDRATE, timeout=DEFAULT_TIMEOUT
        )
        assert reader.descriptor_for_LLM == "Heart Rate and Grip Strength"

    @patch("inputs.plugins.serial_reader.serial.Serial")
    def test_init_with_custom_config(self, mock_serial):
        """Test initialization with custom config."""
        config = SerialReaderConfig(
            port="/dev/ttyACM0",
            baudrate=115200,
            timeout=0.5,
            descriptor="Temperature Sensor",
        )

        reader = SerialReader(config)

        mock_serial.assert_called_once_with("/dev/ttyACM0", 115200, timeout=0.5)
        assert reader.descriptor_for_LLM == "Temperature Sensor"

    @patch("inputs.plugins.serial_reader.serial.Serial")
    def test_init_handles_connection_error(self, mock_serial):
        """Test that connection errors are handled gracefully."""
        import serial

        mock_serial.side_effect = serial.SerialException("Port not found")
        config = SerialReaderConfig()

        reader = SerialReader(config)

        assert reader.ser is None

    @patch("inputs.plugins.serial_reader.serial.Serial")
    @pytest.mark.asyncio
    async def test_poll_returns_none_when_not_connected(self, mock_serial):
        """Test that _poll returns None when serial is not connected."""
        import serial

        mock_serial.side_effect = serial.SerialException("Port not found")
        config = SerialReaderConfig()
        reader = SerialReader(config)

        result = await reader._poll()

        assert result is None

    @patch("inputs.plugins.serial_reader.serial.Serial")
    @pytest.mark.asyncio
    async def test_poll_returns_data(self, mock_serial):
        """Test that _poll returns serial data."""
        mock_ser = MagicMock()
        mock_ser.readline.return_value = b"Pulse: Elevated\n"
        mock_serial.return_value = mock_ser

        config = SerialReaderConfig()
        reader = SerialReader(config)

        result = await reader._poll()

        assert result == "Pulse: Elevated"

    @patch("inputs.plugins.serial_reader.serial.Serial")
    @pytest.mark.asyncio
    async def test_raw_to_text_pulse(self, mock_serial):
        """Test raw_to_text conversion for pulse data."""
        config = SerialReaderConfig()
        reader = SerialReader(config)

        message = await reader._raw_to_text("Pulse: Elevated")

        assert message is not None
        assert "pulse rate" in message.message.lower()
        assert "Elevated" in message.message

    @patch("inputs.plugins.serial_reader.serial.Serial")
    @pytest.mark.asyncio
    async def test_raw_to_text_grip(self, mock_serial):
        """Test raw_to_text conversion for grip data."""
        config = SerialReaderConfig()
        reader = SerialReader(config)

        message = await reader._raw_to_text("Grip: Normal")

        assert message is not None
        assert "grip strength" in message.message.lower()
        assert "Normal" in message.message

    @patch("inputs.plugins.serial_reader.serial.Serial")
    @pytest.mark.asyncio
    async def test_raw_to_text_unknown(self, mock_serial):
        """Test raw_to_text conversion for unknown data."""
        config = SerialReaderConfig()
        reader = SerialReader(config)

        message = await reader._raw_to_text("Unknown: Data")

        assert message is not None
        assert message.message == "No serial data."

    @patch("inputs.plugins.serial_reader.serial.Serial")
    @pytest.mark.asyncio
    async def test_raw_to_text_none(self, mock_serial):
        """Test raw_to_text returns None for None input."""
        config = SerialReaderConfig()
        reader = SerialReader(config)

        message = await reader._raw_to_text(None)

        assert message is None
