from unittest.mock import MagicMock, patch

import pytest

from providers.health_monitor_provider import HealthMonitorProvider
from providers.ub_tts_provider import UbTtsProvider


@pytest.fixture(autouse=True)
def reset_singleton():
    """Reset singleton instances between tests."""
    HealthMonitorProvider.reset()  # type: ignore
    yield
    HealthMonitorProvider.reset()  # type: ignore


@pytest.fixture
def mock_requests():
    with patch("providers.ub_tts_provider.requests") as mock:
        yield mock


def test_initialization():
    """Test UbTtsProvider initialization."""
    provider = UbTtsProvider(url="http://test.url/tts")
    assert provider.tts_url == "http://test.url/tts"
    assert provider.headers == {"Content-Type": "application/json"}


def test_speak_success(mock_requests):
    """Test speak method on success."""
    mock_response = MagicMock()
    mock_response.json.return_value = {"code": 0}
    mock_requests.put.return_value = mock_response

    provider = UbTtsProvider(url="http://test.url/tts")
    result = provider.speak("Hello world")

    assert result is True
    mock_requests.put.assert_called_once()


def test_speak_failure(mock_requests):
    """Test speak method on failure."""
    mock_response = MagicMock()
    mock_response.json.return_value = {"code": 1}
    mock_requests.put.return_value = mock_response

    provider = UbTtsProvider(url="http://test.url/tts")
    result = provider.speak("Hello world")

    assert result is False


def test_speak_exception(mock_requests):
    """Test speak method handles exceptions."""
    import requests

    mock_requests.put.side_effect = requests.exceptions.RequestException("Error")
    mock_requests.exceptions = requests.exceptions

    provider = UbTtsProvider(url="http://test.url/tts")
    result = provider.speak("Hello world")

    assert result is False


def test_recovery_callback_registered():
    """Test that UbTtsProvider registers a recovery callback."""
    provider = UbTtsProvider(url="http://test.url/tts")
    health = HealthMonitorProvider()

    assert "UbTtsProvider" in health._recovery_callbacks
    assert health._recovery_callbacks["UbTtsProvider"] == provider._recover


def test_recover_success(mock_requests):
    """Test that _recover returns True when service is reachable."""
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_requests.get.return_value = mock_response

    provider = UbTtsProvider(url="http://test.url/tts")
    result = provider._recover()

    assert result is True
    mock_requests.get.assert_called_once()


def test_recover_failure_non_200(mock_requests):
    """Test that _recover returns False when service returns non-200."""
    mock_response = MagicMock()
    mock_response.status_code = 500
    mock_requests.get.return_value = mock_response

    provider = UbTtsProvider(url="http://test.url/tts")
    result = provider._recover()

    assert result is False


def test_recover_failure_exception(mock_requests):
    """Test that _recover returns False when an exception occurs."""
    mock_requests.get.side_effect = Exception("Connection error")

    provider = UbTtsProvider(url="http://test.url/tts")
    result = provider._recover()

    assert result is False


def test_get_tts_status_success(mock_requests):
    """Test get_tts_status on success."""
    mock_response = MagicMock()
    mock_response.json.return_value = {"code": 0, "status": "idle"}
    mock_requests.get.return_value = mock_response

    provider = UbTtsProvider(url="http://test.url/tts")
    status = provider.get_tts_status(timestamp=123)

    assert status == "idle"


def test_get_tts_status_error(mock_requests):
    """Test get_tts_status on error."""
    mock_response = MagicMock()
    mock_response.json.return_value = {"code": 1}
    mock_requests.get.return_value = mock_response

    provider = UbTtsProvider(url="http://test.url/tts")
    status = provider.get_tts_status(timestamp=123)

    assert status == "error"
