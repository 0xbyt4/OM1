import sys
from unittest.mock import MagicMock, Mock, patch

import pytest

# Mock modules before importing from providers
mock_om1_speech = MagicMock()
mock_om1_speech.AudioOutputStream = MagicMock()
sys.modules["om1_speech"] = mock_om1_speech

mock_pyaudio = MagicMock()
mock_pyaudio.PyAudio = MagicMock()
mock_instance = MagicMock()
mock_instance.get_default_output_device_info.return_value = {
    "name": "Mock Speaker",
    "index": 0,
}
mock_pyaudio.PyAudio.return_value = mock_instance
sys.modules["pyaudio"] = mock_pyaudio

# Import after mocking
from providers.health_monitor_provider import HealthMonitorProvider  # noqa: E402
from providers.riva_tts_provider import RivaTTSProvider  # noqa: E402


@pytest.fixture(autouse=True)
def reset_singleton():
    """Reset singleton instances between tests."""
    RivaTTSProvider.reset()  # type: ignore
    HealthMonitorProvider.reset()  # type: ignore
    yield
    RivaTTSProvider.reset()  # type: ignore
    HealthMonitorProvider.reset()  # type: ignore


@pytest.fixture(autouse=True)
def mock_audio_stream():
    with patch("providers.riva_tts_provider.AudioOutputStream") as mock:
        mock_instance = MagicMock()
        mock.return_value = mock_instance
        yield mock


def test_initialization(mock_audio_stream):
    provider = RivaTTSProvider(url="test_url")
    assert provider.running is False
    mock_audio_stream.assert_called_once_with(url="test_url", headers=None)


def test_start_stop(mock_audio_stream):
    provider = RivaTTSProvider(url="test_url")
    provider.start()
    assert provider.running is True

    provider.stop()
    assert provider.running is False


def test_register_callback(mock_audio_stream):
    provider = RivaTTSProvider(url="test_url")
    callback = Mock()
    provider.register_tts_state_callback(callback)
    mock_audio_stream.return_value.set_tts_state_callback.assert_called_once_with(
        callback
    )


def test_add_pending_message(mock_audio_stream):
    provider = RivaTTSProvider(url="test_url")
    provider.add_pending_message("test message")
    mock_audio_stream.return_value.add_request.assert_called_once_with(
        {"text": "test message"}
    )


def test_recovery_callback_registered(mock_audio_stream):
    """Test that RivaTTSProvider registers a recovery callback."""
    provider = RivaTTSProvider(url="test_url")
    health = HealthMonitorProvider()

    assert "RivaTTSProvider" in health._recovery_callbacks
    assert health._recovery_callbacks["RivaTTSProvider"] == provider._recover


def test_recover_calls_stop_and_start(mock_audio_stream):
    """Test that _recover stops and restarts the provider."""
    provider = RivaTTSProvider(url="test_url")

    with (
        patch.object(provider, "stop") as mock_stop,
        patch.object(provider, "start") as mock_start,
    ):
        result = provider._recover()

        mock_stop.assert_called_once()
        mock_start.assert_called_once()
        assert result is True


def test_recover_returns_false_on_exception(mock_audio_stream):
    """Test that _recover returns False when an exception occurs."""
    provider = RivaTTSProvider(url="test_url")

    with patch.object(provider, "stop", side_effect=Exception("Test error")):
        result = provider._recover()

        assert result is False
