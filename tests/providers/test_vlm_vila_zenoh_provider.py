from unittest.mock import MagicMock, patch

import pytest

from providers.health_monitor_provider import HealthMonitorProvider
from providers.vlm_vila_zenoh_provider import VLMVilaZenohProvider


@pytest.fixture
def ws_url():
    return "ws://test.url"


@pytest.fixture
def topic():
    return "rgb_image"


@pytest.fixture(autouse=True)
def reset_singleton():
    """Reset singleton instances between tests."""
    VLMVilaZenohProvider.reset()  # type: ignore
    HealthMonitorProvider.reset()  # type: ignore
    yield
    VLMVilaZenohProvider.reset()  # type: ignore
    HealthMonitorProvider.reset()  # type: ignore


@pytest.fixture
def mock_dependencies():
    with (
        patch("providers.vlm_vila_zenoh_provider.ws.Client") as mock_ws_client,
        patch(
            "providers.vlm_vila_zenoh_provider.VideoZenohStream"
        ) as mock_video_stream,
    ):
        yield mock_ws_client, mock_video_stream


def test_initialization(ws_url, topic, mock_dependencies):
    """Test VLMVilaZenohProvider initialization."""
    mock_ws_client, mock_video_stream = mock_dependencies
    provider = VLMVilaZenohProvider(ws_url, topic)

    mock_ws_client.assert_called_once_with(url=ws_url)
    mock_video_stream.assert_called_once()
    assert provider.running is False


def test_singleton_pattern(ws_url, topic, mock_dependencies):
    """Test singleton pattern."""
    provider1 = VLMVilaZenohProvider(ws_url, topic)
    provider2 = VLMVilaZenohProvider(ws_url, topic)
    assert provider1 is provider2


def test_register_message_callback(ws_url, topic, mock_dependencies):
    """Test register_message_callback."""
    mock_ws_client, mock_video_stream = mock_dependencies
    provider = VLMVilaZenohProvider(ws_url, topic)
    callback = MagicMock()
    provider.register_message_callback(callback)

    mock_ws_client.return_value.register_message_callback.assert_called_once_with(
        callback
    )


def test_start(ws_url, topic, mock_dependencies):
    """Test start method."""
    mock_ws_client, mock_video_stream = mock_dependencies
    provider = VLMVilaZenohProvider(ws_url, topic)
    provider.start()

    assert provider.running is True
    mock_ws_client.return_value.start.assert_called_once()
    mock_video_stream.return_value.start.assert_called_once()


def test_stop(ws_url, topic, mock_dependencies):
    """Test stop method."""
    mock_ws_client, mock_video_stream = mock_dependencies
    provider = VLMVilaZenohProvider(ws_url, topic)
    provider.start()
    provider.stop()

    assert provider.running is False
    mock_video_stream.return_value.stop.assert_called_once()
    mock_ws_client.return_value.stop.assert_called_once()


def test_recovery_callback_registered(ws_url, topic, mock_dependencies):
    """Test that VLMVilaZenohProvider registers a recovery callback."""
    provider = VLMVilaZenohProvider(ws_url, topic)
    health = HealthMonitorProvider()

    assert "VLMVilaZenohProvider" in health._recovery_callbacks
    assert health._recovery_callbacks["VLMVilaZenohProvider"] == provider._recover


def test_recover_calls_stop_and_start(ws_url, topic, mock_dependencies):
    """Test that _recover stops and restarts the provider."""
    provider = VLMVilaZenohProvider(ws_url, topic)

    with (
        patch.object(provider, "stop") as mock_stop,
        patch.object(provider, "start") as mock_start,
    ):
        result = provider._recover()

        mock_stop.assert_called_once()
        mock_start.assert_called_once()
        assert result is True


def test_recover_returns_false_on_exception(ws_url, topic, mock_dependencies):
    """Test that _recover returns False when an exception occurs."""
    provider = VLMVilaZenohProvider(ws_url, topic)

    with patch.object(provider, "stop", side_effect=Exception("Test error")):
        result = provider._recover()

        assert result is False
