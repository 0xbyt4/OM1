from unittest.mock import MagicMock, patch

import pytest

from providers.health_monitor_provider import HealthMonitorProvider
from providers.vlm_openai_rtsp_provider import VLMOpenAIRTSPProvider


@pytest.fixture
def base_url():
    return "https://api.openmind.org/api/core/openai"


@pytest.fixture
def api_key():
    return "test_api_key"


@pytest.fixture
def rtsp_url():
    return "rtsp://localhost:8554/top_camera"


@pytest.fixture(autouse=True)
def reset_singleton():
    """Reset singleton instances between tests."""
    VLMOpenAIRTSPProvider.reset()  # type: ignore
    HealthMonitorProvider.reset()  # type: ignore
    yield
    VLMOpenAIRTSPProvider.reset()  # type: ignore
    HealthMonitorProvider.reset()  # type: ignore


@pytest.fixture
def mock_dependencies():
    with (
        patch("providers.vlm_openai_rtsp_provider.AsyncOpenAI") as mock_client,
        patch(
            "providers.vlm_openai_rtsp_provider.VideoRTSPStream"
        ) as mock_video_stream,
    ):
        yield mock_client, mock_video_stream


def test_initialization(base_url, api_key, rtsp_url, mock_dependencies):
    """Test VLMOpenAIRTSPProvider initialization."""
    mock_client, mock_video_stream = mock_dependencies
    provider = VLMOpenAIRTSPProvider(base_url, api_key, rtsp_url)

    mock_client.assert_called_once_with(api_key=api_key, base_url=base_url)
    mock_video_stream.assert_called_once()
    assert provider.running is False


def test_singleton_pattern(base_url, api_key, rtsp_url, mock_dependencies):
    """Test singleton pattern."""
    provider1 = VLMOpenAIRTSPProvider(base_url, api_key, rtsp_url)
    provider2 = VLMOpenAIRTSPProvider(base_url, api_key, rtsp_url)
    assert provider1 is provider2


def test_register_message_callback(base_url, api_key, rtsp_url, mock_dependencies):
    """Test register_message_callback."""
    provider = VLMOpenAIRTSPProvider(base_url, api_key, rtsp_url)
    callback = MagicMock()
    provider.register_message_callback(callback)

    assert provider.message_callback == callback


def test_start(base_url, api_key, rtsp_url, mock_dependencies):
    """Test start method."""
    mock_client, mock_video_stream = mock_dependencies
    provider = VLMOpenAIRTSPProvider(base_url, api_key, rtsp_url)

    with patch("asyncio.create_task") as mock_create_task:
        provider.start()

        assert provider.running is True
        mock_video_stream.return_value.start.assert_called_once()
        mock_create_task.assert_called_once()


def test_stop(base_url, api_key, rtsp_url, mock_dependencies):
    """Test stop method."""
    mock_client, mock_video_stream = mock_dependencies
    provider = VLMOpenAIRTSPProvider(base_url, api_key, rtsp_url)

    with patch("asyncio.create_task"):
        provider.start()
        provider.stop()

        assert provider.running is False
        mock_video_stream.return_value.stop.assert_called_once()


def test_recovery_callback_registered(base_url, api_key, rtsp_url, mock_dependencies):
    """Test that VLMOpenAIRTSPProvider registers a recovery callback."""
    provider = VLMOpenAIRTSPProvider(base_url, api_key, rtsp_url)
    health = HealthMonitorProvider()

    assert "VLMOpenAIRTSPProvider" in health._recovery_callbacks
    assert health._recovery_callbacks["VLMOpenAIRTSPProvider"] == provider._recover


def test_recover_calls_stop_and_start(base_url, api_key, rtsp_url, mock_dependencies):
    """Test that _recover stops and restarts the provider."""
    provider = VLMOpenAIRTSPProvider(base_url, api_key, rtsp_url)

    with (
        patch.object(provider, "stop") as mock_stop,
        patch.object(provider, "start") as mock_start,
    ):
        result = provider._recover()

        mock_stop.assert_called_once()
        mock_start.assert_called_once()
        assert result is True


def test_recover_returns_false_on_exception(
    base_url, api_key, rtsp_url, mock_dependencies
):
    """Test that _recover returns False when an exception occurs."""
    provider = VLMOpenAIRTSPProvider(base_url, api_key, rtsp_url)

    with patch.object(provider, "stop", side_effect=Exception("Test error")):
        result = provider._recover()

        assert result is False
