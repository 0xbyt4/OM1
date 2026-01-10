from unittest.mock import MagicMock, patch

import pytest

from providers.health_monitor_provider import HealthMonitorProvider
from providers.ubtech_asr_provider import UbtechASRProvider


@pytest.fixture
def robot_ip():
    return "192.168.1.100"


@pytest.fixture(autouse=True)
def reset_singleton():
    """Reset singleton instances between tests."""
    UbtechASRProvider.reset()  # type: ignore
    HealthMonitorProvider.reset()  # type: ignore
    yield
    UbtechASRProvider.reset()  # type: ignore
    HealthMonitorProvider.reset()  # type: ignore


@pytest.fixture
def mock_requests():
    with patch("providers.ubtech_asr_provider.requests") as mock:
        mock_session = MagicMock()
        mock.Session.return_value = mock_session
        yield mock, mock_session


def test_initialization(robot_ip, mock_requests):
    """Test UbtechASRProvider initialization."""
    mock, mock_session = mock_requests
    provider = UbtechASRProvider(robot_ip)

    assert provider.robot_ip == robot_ip
    assert provider.running is False
    assert provider.basic_url == f"http://{robot_ip}:9090/v1/"


def test_singleton_pattern(robot_ip, mock_requests):
    """Test singleton pattern."""
    provider1 = UbtechASRProvider(robot_ip)
    provider2 = UbtechASRProvider(robot_ip)
    assert provider1 is provider2


def test_register_message_callback(robot_ip, mock_requests):
    """Test register_message_callback."""
    provider = UbtechASRProvider(robot_ip)
    callback = MagicMock()
    provider.register_message_callback(callback)

    assert provider._message_callback == callback


def test_start(robot_ip, mock_requests):
    """Test start method."""
    provider = UbtechASRProvider(robot_ip)

    with patch.object(provider, "_run"):
        provider.start()

        assert provider.running is True
        assert provider._thread is not None


def test_stop(robot_ip, mock_requests):
    """Test stop method."""
    mock, mock_session = mock_requests
    provider = UbtechASRProvider(robot_ip)

    with patch.object(provider, "_run"):
        provider.start()
        provider.stop()

        assert provider.running is False


def test_pause_resume(robot_ip, mock_requests):
    """Test pause and resume methods."""
    provider = UbtechASRProvider(robot_ip)

    assert provider.paused is False

    provider.pause()
    assert provider.paused is True

    provider.resume()
    assert provider.paused is False
    assert provider.just_resumed is True


def test_recovery_callback_registered(robot_ip, mock_requests):
    """Test that UbtechASRProvider registers a recovery callback."""
    provider = UbtechASRProvider(robot_ip)
    health = HealthMonitorProvider()

    assert "UbtechASRProvider" in health._recovery_callbacks
    assert health._recovery_callbacks["UbtechASRProvider"] == provider._recover


def test_recover_calls_stop_and_start(robot_ip, mock_requests):
    """Test that _recover stops and restarts the provider."""
    provider = UbtechASRProvider(robot_ip)

    with (
        patch.object(provider, "stop") as mock_stop,
        patch.object(provider, "start") as mock_start,
    ):
        result = provider._recover()

        mock_stop.assert_called_once()
        mock_start.assert_called_once()
        assert result is True


def test_recover_returns_false_on_exception(robot_ip, mock_requests):
    """Test that _recover returns False when an exception occurs."""
    provider = UbtechASRProvider(robot_ip)

    with patch.object(provider, "stop", side_effect=Exception("Test error")):
        result = provider._recover()

        assert result is False
