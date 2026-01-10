from unittest.mock import MagicMock, patch

import pytest

from providers.health_monitor_provider import HealthMonitorProvider
from providers.rtk_provider import RtkProvider


@pytest.fixture
def serial_port():
    return "/dev/ttyUSB0"


@pytest.fixture(autouse=True)
def reset_singleton():
    """Reset singleton instances between tests."""
    RtkProvider.reset()  # type: ignore
    HealthMonitorProvider.reset()  # type: ignore
    yield
    RtkProvider.reset()  # type: ignore
    HealthMonitorProvider.reset()  # type: ignore


@pytest.fixture
def mock_serial():
    with patch("providers.rtk_provider.serial.Serial") as mock:
        mock_instance = MagicMock()
        mock.return_value = mock_instance
        yield mock, mock_instance


def test_initialization(serial_port, mock_serial):
    """Test RtkProvider initialization."""
    mock, mock_instance = mock_serial
    provider = RtkProvider(serial_port)

    assert provider.running is True
    assert provider.lat == 0.0
    assert provider.lon == 0.0


def test_singleton_pattern(serial_port, mock_serial):
    """Test singleton pattern."""
    provider1 = RtkProvider(serial_port)
    provider2 = RtkProvider(serial_port)
    assert provider1 is provider2


def test_start(serial_port, mock_serial):
    """Test start method."""
    provider = RtkProvider(serial_port)

    assert provider.running is True
    assert provider._thread is not None


def test_stop(serial_port, mock_serial):
    """Test stop method."""
    provider = RtkProvider(serial_port)
    provider.stop()

    assert provider.running is False


def test_recovery_callback_registered(serial_port, mock_serial):
    """Test that RtkProvider registers a recovery callback."""
    provider = RtkProvider(serial_port)
    health = HealthMonitorProvider()

    assert "RtkProvider" in health._recovery_callbacks
    assert health._recovery_callbacks["RtkProvider"] == provider._recover


def test_recover_calls_stop_and_start(serial_port, mock_serial):
    """Test that _recover stops and restarts the provider."""
    provider = RtkProvider(serial_port)

    with (
        patch.object(provider, "stop") as mock_stop,
        patch.object(provider, "start") as mock_start,
    ):
        result = provider._recover()

        mock_stop.assert_called_once()
        mock_start.assert_called_once()
        assert result is True


def test_recover_returns_false_on_exception(serial_port, mock_serial):
    """Test that _recover returns False when an exception occurs."""
    provider = RtkProvider(serial_port)

    with patch.object(provider, "stop", side_effect=Exception("Test error")):
        result = provider._recover()

        assert result is False


def test_data_property(serial_port, mock_serial):
    """Test data property returns RTK data."""
    provider = RtkProvider(serial_port)

    # Initially should be None or have default values
    provider._rtk = {
        "rtk_lat": 37.7749,
        "rtk_lon": -122.4194,
        "rtk_alt": 10.0,
        "rtk_sat": 10,
        "rtk_qua": 4,
        "rtk_unix_ts": 1234567890.0,
    }

    data = provider.data
    assert data["rtk_lat"] == 37.7749
    assert data["rtk_lon"] == -122.4194
