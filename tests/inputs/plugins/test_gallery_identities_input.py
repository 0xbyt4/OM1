import logging
from queue import Full, Queue
from unittest.mock import MagicMock, patch


class TestGalleryIdentitiesMessageHandler:
    def test_logs_warning_when_retry_put_fails(self, caplog):
        """
        Test that a warning is logged when queue retry put fails.

        This test verifies that when the message buffer is full and
        the retry put also fails, a warning is logged instead of
        silently passing.
        """
        with (
            patch(
                "inputs.plugins.gallery_identities_input.GalleryIdentitiesProvider"
            ) as mock_provider_class,
            patch("inputs.plugins.gallery_identities_input.IOProvider"),
        ):
            mock_provider = MagicMock()
            mock_provider_class.return_value = mock_provider

            from inputs.plugins.gallery_identities_input import (
                GalleryIdentities,
                GalleryIdentitiesConfig,
            )

            config = GalleryIdentitiesConfig()
            input_handler = GalleryIdentities(config)

            mock_queue = MagicMock(spec=Queue)
            mock_queue.put_nowait.side_effect = Full()
            mock_queue.get_nowait.return_value = "old_message"
            input_handler.message_buffer = mock_queue

            with caplog.at_level(logging.WARNING):
                input_handler._handle_gallery_message("test_message")

            assert any(
                "Failed to enqueue" in record.message for record in caplog.records
            ), "Expected warning log when retry put fails"

    def test_normal_put_succeeds_without_warning(self, caplog):
        """
        Test that no warning is logged when queue put succeeds.
        """
        with (
            patch(
                "inputs.plugins.gallery_identities_input.GalleryIdentitiesProvider"
            ) as mock_provider_class,
            patch("inputs.plugins.gallery_identities_input.IOProvider"),
        ):
            mock_provider = MagicMock()
            mock_provider_class.return_value = mock_provider

            from inputs.plugins.gallery_identities_input import (
                GalleryIdentities,
                GalleryIdentitiesConfig,
            )

            config = GalleryIdentitiesConfig()
            input_handler = GalleryIdentities(config)

            mock_queue = MagicMock(spec=Queue)
            mock_queue.put_nowait.return_value = None
            input_handler.message_buffer = mock_queue

            with caplog.at_level(logging.WARNING):
                input_handler._handle_gallery_message("test_message")

            warning_logs = [r for r in caplog.records if r.levelno >= logging.WARNING]
            assert len(warning_logs) == 0, "No warning expected for successful put"
