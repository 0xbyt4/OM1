from unittest.mock import MagicMock, patch

import pytest

from providers.health_monitor_provider import HealthMonitorProvider
from providers.vlm_vila_provider import VLMVilaProvider


@pytest.fixture
def ws_url():
    return "ws://test.url"


@pytest.fixture
def fps():
    return 30


@pytest.fixture(autouse=True)
def reset_singleton():
    """Reset singleton instances between tests."""
    VLMVilaProvider.reset()  # type: ignore
    HealthMonitorProvider.reset()  # type: ignore
    yield
    VLMVilaProvider.reset()  # type: ignore
    HealthMonitorProvider.reset()  # type: ignore


@pytest.fixture
def mock_dependencies():
    mock_ws_client_instance = MagicMock()
    mock_video_stream_instance = MagicMock()
    with (
        patch(
            "providers.vlm_vila_provider.ws.Client",
            return_value=mock_ws_client_instance,
        ) as mock_ws_client_class,
        patch(
            "providers.vlm_vila_provider.VideoStream",
            return_value=mock_video_stream_instance,
        ) as mock_video_stream_class,
    ):
        yield mock_ws_client_class, mock_video_stream_class, mock_ws_client_instance, mock_video_stream_instance


def test_initialization(ws_url, fps, mock_dependencies):
    (
        mock_ws_client_class,
        mock_video_stream_class,
        mock_ws_client_instance,
        mock_video_stream_instance,
    ) = mock_dependencies
    provider = VLMVilaProvider(ws_url, fps=fps)

    mock_ws_client_class.assert_called_once_with(url=ws_url)
    mock_video_stream_class.assert_called_once_with(
        mock_ws_client_instance.send_message, fps=fps, device_index=0
    )

    assert not provider.running
    assert provider.ws_client is mock_ws_client_instance
    assert provider.video_stream is mock_video_stream_instance


def test_singleton_pattern(ws_url, fps, mock_dependencies):
    provider1 = VLMVilaProvider(ws_url, fps=fps)
    provider2 = VLMVilaProvider(ws_url, fps=fps)

    assert provider1 is provider2
    assert provider1.ws_client is provider2.ws_client
    assert provider1.video_stream is provider2.video_stream


def test_register_message_callback(ws_url, fps, mock_dependencies):
    _, _, mock_ws_client_instance, _ = mock_dependencies
    provider = VLMVilaProvider(ws_url, fps=fps)
    callback = MagicMock()

    provider.register_message_callback(callback)
    mock_ws_client_instance.register_message_callback.assert_called_once_with(callback)


def test_start(ws_url, fps, mock_dependencies):
    _, _, mock_ws_client_instance, mock_video_stream_instance = mock_dependencies
    provider = VLMVilaProvider(ws_url, fps=fps)
    provider.start()

    assert provider.running
    mock_ws_client_instance.start.assert_called_once()
    mock_video_stream_instance.start.assert_called_once()


def test_stop(ws_url, fps, mock_dependencies):
    _, _, mock_ws_client_instance, mock_video_stream_instance = mock_dependencies
    provider = VLMVilaProvider(ws_url, fps=fps)
    provider.start()
    provider.stop()

    assert not provider.running
    mock_video_stream_instance.stop.assert_called_once()
    mock_ws_client_instance.stop.assert_called_once()


def test_recovery_callback_registered(ws_url, fps, mock_dependencies):
    """Test that VLMVilaProvider registers a recovery callback."""
    provider = VLMVilaProvider(ws_url, fps=fps)
    health = HealthMonitorProvider()

    assert "VLMVilaProvider" in health._recovery_callbacks
    assert health._recovery_callbacks["VLMVilaProvider"] == provider._recover


def test_recover_calls_stop_and_start(ws_url, fps, mock_dependencies):
    """Test that _recover stops and restarts the provider."""
    provider = VLMVilaProvider(ws_url, fps=fps)

    with (
        patch.object(provider, "stop") as mock_stop,
        patch.object(provider, "start") as mock_start,
    ):
        result = provider._recover()

        mock_stop.assert_called_once()
        mock_start.assert_called_once()
        assert result is True


def test_recover_returns_false_on_exception(ws_url, fps, mock_dependencies):
    """Test that _recover returns False when an exception occurs."""
    provider = VLMVilaProvider(ws_url, fps=fps)

    with patch.object(provider, "stop", side_effect=Exception("Test error")):
        result = provider._recover()

        assert result is False
