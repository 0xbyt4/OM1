"""Tests for serial_reader.py malformed input handling.

These tests verify that _raw_to_text handles malformed input gracefully
without raising IndexError.

Bug fix for: src/inputs/plugins/serial_reader.py:138-149
"""

import pytest


class TestSerialReaderMalformedInput:
    """Tests for SerialReader handling of malformed input."""

    @pytest.mark.asyncio
    async def test_pulse_without_value_handles_gracefully(self):
        """
        Test that 'Pulse:' without value is handled gracefully.

        Previously this raised IndexError. Now it should return a message
        indicating the value is missing.
        """
        from unittest.mock import patch

        with patch("inputs.plugins.serial_reader.serial"):
            from inputs.base import SensorConfig
            from inputs.plugins.serial_reader import SerialReader

            config = SensorConfig(input_name="test")
            reader = SerialReader(config)

            result = await reader._raw_to_text("Pulse:")
            assert result is not None
            assert "missing" in result.message.lower()

    @pytest.mark.asyncio
    async def test_grip_without_value_handles_gracefully(self):
        """
        Test that 'Grip:' without value is handled gracefully.

        Previously this raised IndexError. Now it should return a message
        indicating the value is missing.
        """
        from unittest.mock import patch

        with patch("inputs.plugins.serial_reader.serial"):
            from inputs.base import SensorConfig
            from inputs.plugins.serial_reader import SerialReader

            config = SensorConfig(input_name="test")
            reader = SerialReader(config)

            result = await reader._raw_to_text("Grip:")
            assert result is not None
            assert "missing" in result.message.lower()

    @pytest.mark.asyncio
    async def test_pulse_with_only_spaces_handles_gracefully(self):
        """
        Test that 'Pulse: ' (with trailing spaces but no value) is handled.
        """
        from unittest.mock import patch

        with patch("inputs.plugins.serial_reader.serial"):
            from inputs.base import SensorConfig
            from inputs.plugins.serial_reader import SerialReader

            config = SensorConfig(input_name="test")
            reader = SerialReader(config)

            result = await reader._raw_to_text("Pulse:   ")
            assert result is not None
            assert "missing" in result.message.lower()

    @pytest.mark.asyncio
    async def test_pulse_with_valid_value_works(self):
        """Test that valid 'Pulse: Elevated' input works correctly."""
        from unittest.mock import patch

        with patch("inputs.plugins.serial_reader.serial"):
            from inputs.base import SensorConfig
            from inputs.plugins.serial_reader import SerialReader

            config = SensorConfig(input_name="test")
            reader = SerialReader(config)

            result = await reader._raw_to_text("Pulse: Elevated")
            assert result is not None
            assert "Elevated" in result.message
            assert "pulse rate" in result.message.lower()

    @pytest.mark.asyncio
    async def test_grip_with_valid_value_works(self):
        """Test that valid 'Grip: Normal' input works correctly."""
        from unittest.mock import patch

        with patch("inputs.plugins.serial_reader.serial"):
            from inputs.base import SensorConfig
            from inputs.plugins.serial_reader import SerialReader

            config = SensorConfig(input_name="test")
            reader = SerialReader(config)

            result = await reader._raw_to_text("Grip: Normal")
            assert result is not None
            assert "Normal" in result.message
            assert "grip strength" in result.message.lower()

    @pytest.mark.asyncio
    async def test_unknown_input_returns_no_serial_data(self):
        """Test that unknown input returns 'No serial data' message."""
        from unittest.mock import patch

        with patch("inputs.plugins.serial_reader.serial"):
            from inputs.base import SensorConfig
            from inputs.plugins.serial_reader import SerialReader

            config = SensorConfig(input_name="test")
            reader = SerialReader(config)

            result = await reader._raw_to_text("Unknown: Data")
            assert result is not None
            assert "No serial data" in result.message
