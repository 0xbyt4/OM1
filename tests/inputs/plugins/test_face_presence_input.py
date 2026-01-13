"""Tests for face_presence_input module."""

import logging
from queue import Full, Queue
from unittest.mock import MagicMock, patch


class TestFacePresenceMessageHandler:
    """Tests for _handle_face_message method logging behavior."""

    def test_logs_warning_when_retry_put_fails(self, caplog):
        """
        Test that a warning is logged when queue retry put fails.

        This test verifies that when the message buffer is full and
        the retry put also fails, a warning is logged instead of
        silently passing.
        """
        # Mock dependencies to avoid actual provider initialization
        with (
            patch(
                "inputs.plugins.face_presence_input.FacePresenceProvider"
            ) as mock_provider_class,
            patch("inputs.plugins.face_presence_input.IOProvider"),
        ):
            mock_provider = MagicMock()
            mock_provider_class.return_value = mock_provider

            from inputs.plugins.face_presence_input import (
                FacePresence,
                FacePresenceConfig,
            )

            config = FacePresenceConfig()
            input_handler = FacePresence(config)

            # Replace the message_buffer with a mock that always raises Full
            mock_queue = MagicMock(spec=Queue)
            mock_queue.put_nowait.side_effect = Full()
            mock_queue.get_nowait.return_value = "old_message"
            input_handler.message_buffer = mock_queue

            # Call the handler which should trigger the retry logic
            with caplog.at_level(logging.WARNING):
                input_handler._handle_face_message("test_message")

            # Verify warning was logged for the failed retry
            assert any(
                "Failed to enqueue" in record.message for record in caplog.records
            ), "Expected warning log when retry put fails"

    def test_normal_put_succeeds_without_warning(self, caplog):
        """
        Test that no warning is logged when queue put succeeds.
        """
        with (
            patch(
                "inputs.plugins.face_presence_input.FacePresenceProvider"
            ) as mock_provider_class,
            patch("inputs.plugins.face_presence_input.IOProvider"),
        ):
            mock_provider = MagicMock()
            mock_provider_class.return_value = mock_provider

            from inputs.plugins.face_presence_input import (
                FacePresence,
                FacePresenceConfig,
            )

            config = FacePresenceConfig()
            input_handler = FacePresence(config)

            # Replace with a mock that succeeds
            mock_queue = MagicMock(spec=Queue)
            mock_queue.put_nowait.return_value = None  # Success
            input_handler.message_buffer = mock_queue

            with caplog.at_level(logging.WARNING):
                input_handler._handle_face_message("test_message")

            # No warnings should be logged
            warning_logs = [r for r in caplog.records if r.levelno >= logging.WARNING]
            assert len(warning_logs) == 0, "No warning expected for successful put"
