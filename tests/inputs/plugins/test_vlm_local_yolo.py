"""
Tests for VLM_Local_YOLO input plugin.

Specifically tests for resource management (VideoCapture release).
"""

from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pytest

from inputs.plugins.vlm_local_yolo import VLM_Local_YOLO, VLM_Local_YOLOConfig


@pytest.fixture
def mock_yolo_model():
    """Mock YOLO model to avoid loading actual model."""
    with patch("inputs.plugins.vlm_local_yolo.YOLO") as mock:
        mock_instance = Mock()
        mock.return_value = mock_instance
        yield mock_instance


@pytest.fixture
def mock_check_webcam():
    """Mock webcam check to return valid resolution."""
    with patch(
        "inputs.plugins.vlm_local_yolo.check_webcam", return_value=(640, 480)
    ) as mock:
        yield mock


@pytest.fixture
def mock_cv2_video_capture():
    """Mock cv2.VideoCapture to avoid actual camera access."""
    with patch("inputs.plugins.vlm_local_yolo.cv2.VideoCapture") as mock:
        mock_instance = MagicMock()
        dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        mock_instance.read.return_value = (True, dummy_frame)
        mock_instance.set.return_value = True
        mock_instance.release.return_value = None
        mock.return_value = mock_instance
        yield mock_instance


@pytest.fixture
def mock_odom_provider():
    """Mock OdomProvider."""
    with patch("inputs.plugins.vlm_local_yolo.OdomProvider") as mock:
        mock_instance = Mock()
        mock.return_value = mock_instance
        yield mock_instance


@pytest.fixture
def mock_io_provider():
    """Mock IOProvider."""
    with patch("inputs.plugins.vlm_local_yolo.IOProvider") as mock:
        mock_instance = Mock()
        mock.return_value = mock_instance
        yield mock_instance


@pytest.fixture
def vlm_local_yolo(
    mock_yolo_model,
    mock_check_webcam,
    mock_cv2_video_capture,
    mock_odom_provider,
    mock_io_provider,
):
    """Create VLM_Local_YOLO instance with all dependencies mocked."""
    config = VLM_Local_YOLOConfig(camera_index=0)
    return VLM_Local_YOLO(config=config)


class TestVideoCaptureLeak:
    """Tests for VideoCapture resource management (BUG-001)."""

    def test_stop_releases_video_capture(
        self,
        mock_yolo_model,
        mock_check_webcam,
        mock_cv2_video_capture,
        mock_odom_provider,
        mock_io_provider,
    ):
        """
        Test that calling stop() releases the VideoCapture resource.

        This test verifies the fix for BUG-001: VideoCapture not being released.
        Without proper cleanup, the camera device remains locked after use.
        """
        config = VLM_Local_YOLOConfig(camera_index=0)
        vlm = VLM_Local_YOLO(config=config)

        assert vlm.cap is not None, "VideoCapture should be initialized"

        vlm.stop()

        mock_cv2_video_capture.release.assert_called_once()

    def test_stop_handles_none_capture(
        self,
        mock_yolo_model,
        mock_check_webcam,
        mock_odom_provider,
        mock_io_provider,
    ):
        """
        Test that stop() handles case when cap is None (no camera).
        """
        with patch("inputs.plugins.vlm_local_yolo.check_webcam", return_value=(0, 0)):
            with patch("inputs.plugins.vlm_local_yolo.cv2.VideoCapture") as mock_cap:
                config = VLM_Local_YOLOConfig(camera_index=0)
                vlm = VLM_Local_YOLO(config=config)

                vlm.stop()

                mock_cap.return_value.release.assert_not_called()

    def test_destructor_releases_video_capture(
        self,
        mock_yolo_model,
        mock_check_webcam,
        mock_cv2_video_capture,
        mock_odom_provider,
        mock_io_provider,
    ):
        """
        Test that __del__ releases VideoCapture when object is garbage collected.
        """
        config = VLM_Local_YOLOConfig(camera_index=0)
        vlm = VLM_Local_YOLO(config=config)

        assert vlm.cap is not None

        del vlm

        mock_cv2_video_capture.release.assert_called()


class TestVLMLocalYOLOBasic:
    """Basic functionality tests for VLM_Local_YOLO."""

    def test_initialization(self, vlm_local_yolo):
        """Test that VLM_Local_YOLO initializes correctly."""
        assert vlm_local_yolo.cap is not None
        assert vlm_local_yolo.have_cam is True

    @pytest.mark.asyncio
    async def test_poll_calls_camera_read(self, vlm_local_yolo, mock_cv2_video_capture):
        """Test that _poll reads from camera when available."""
        with patch.object(vlm_local_yolo, "model") as mock_model:
            mock_model.return_value = []
            try:
                await vlm_local_yolo._poll()
            except Exception:
                pass
            mock_cv2_video_capture.read.assert_called()

    def test_initialization_without_camera(
        self,
        mock_yolo_model,
        mock_odom_provider,
        mock_io_provider,
    ):
        """Test initialization when no camera is available."""
        with patch("inputs.plugins.vlm_local_yolo.check_webcam", return_value=(0, 0)):
            config = VLM_Local_YOLOConfig(camera_index=0)
            vlm = VLM_Local_YOLO(config=config)

            assert vlm.have_cam is False
            assert vlm.cap is None
