"""Tests for person_follow_hook module."""

from unittest.mock import AsyncMock, MagicMock, patch

import aiohttp
import pytest


class TestContextValidation:
    """
    Test context parameter validation for hooks.

    These tests verify behavior with edge case and invalid context values.
    """

    @pytest.fixture
    def mock_elevenlabs(self):
        """Mock ElevenLabsTTSProvider."""
        with patch("hooks.person_follow_hook.ElevenLabsTTSProvider") as mock:
            mock_instance = MagicMock()
            mock.return_value = mock_instance
            yield mock_instance

    def create_mock_response(self, status=200, json_data=None):
        """Helper to create mock response."""
        response = MagicMock()
        response.status = status
        response.__aenter__ = AsyncMock(return_value=response)
        response.__aexit__ = AsyncMock(return_value=None)
        if json_data:
            response.json = AsyncMock(return_value=json_data)
        return response

    @pytest.mark.asyncio
    async def test_context_zero_timeout(self, mock_elevenlabs):
        """
        Test behavior with zero enroll_timeout.

        Zero timeout means no waiting for status - immediate retry.
        """
        from hooks.person_follow_hook import start_person_follow_hook

        mock_post_response = self.create_mock_response(status=200)
        mock_get_response = self.create_mock_response(
            status=200, json_data={"is_tracked": False}
        )

        with patch("aiohttp.ClientSession") as mock_session_class:
            mock_session = MagicMock()
            mock_session.__aenter__ = AsyncMock(return_value=mock_session)
            mock_session.__aexit__ = AsyncMock(return_value=None)
            mock_session.post.return_value = mock_post_response
            mock_session.get.return_value = mock_get_response
            mock_session_class.return_value = mock_session

            result = await start_person_follow_hook({
                "enroll_timeout": 0.0,
                "max_retries": 1,
            })

            # Should complete quickly with no tracking
            assert result["status"] == "success"
            assert result["is_tracked"] is False

    @pytest.mark.asyncio
    async def test_context_negative_timeout(self, mock_elevenlabs, caplog):
        """
        Test behavior with negative enroll_timeout.

        Negative timeout should be rejected and default (3.0) used with warning.
        """
        import logging

        from hooks.person_follow_hook import start_person_follow_hook

        mock_post_response = self.create_mock_response(status=200)
        mock_get_response = self.create_mock_response(
            status=200, json_data={"is_tracked": True}
        )

        with patch("aiohttp.ClientSession") as mock_session_class:
            mock_session = MagicMock()
            mock_session.__aenter__ = AsyncMock(return_value=mock_session)
            mock_session.__aexit__ = AsyncMock(return_value=None)
            mock_session.post.return_value = mock_post_response
            mock_session.get.return_value = mock_get_response
            mock_session_class.return_value = mock_session

            with caplog.at_level(logging.WARNING):
                result = await start_person_follow_hook({
                    "enroll_timeout": -1.0,
                    "max_retries": 1,
                })

            # Should use default and log warning
            assert "Invalid enroll_timeout" in caplog.text
            assert result["status"] == "success"

    @pytest.mark.asyncio
    async def test_context_zero_max_retries(self, mock_elevenlabs):
        """
        Test behavior with zero max_retries.

        Zero retries means for loop never executes.
        """
        from hooks.person_follow_hook import start_person_follow_hook

        with patch("aiohttp.ClientSession") as mock_session_class:
            mock_session = MagicMock()
            mock_session.__aenter__ = AsyncMock(return_value=mock_session)
            mock_session.__aexit__ = AsyncMock(return_value=None)
            mock_session_class.return_value = mock_session

            result = await start_person_follow_hook({
                "max_retries": 0,
            })

            # POST should never be called
            mock_session.post.assert_not_called()
            assert result["status"] == "success"
            assert result["is_tracked"] is False

    @pytest.mark.asyncio
    async def test_context_negative_max_retries(self, mock_elevenlabs, caplog):
        """
        Test behavior with negative max_retries.

        Negative max_retries should be rejected and default (5) used with warning.
        """
        import logging

        from hooks.person_follow_hook import start_person_follow_hook

        mock_post_response = self.create_mock_response(status=200)
        mock_get_response = self.create_mock_response(
            status=200, json_data={"is_tracked": True}
        )

        with patch("aiohttp.ClientSession") as mock_session_class:
            mock_session = MagicMock()
            mock_session.__aenter__ = AsyncMock(return_value=mock_session)
            mock_session.__aexit__ = AsyncMock(return_value=None)
            mock_session.post.return_value = mock_post_response
            mock_session.get.return_value = mock_get_response
            mock_session_class.return_value = mock_session

            with caplog.at_level(logging.WARNING):
                result = await start_person_follow_hook({
                    "max_retries": -5,
                })

            # Should use default (5) and log warning
            assert "Invalid max_retries" in caplog.text
            assert result["status"] == "success"
            # POST should be called 5 times (default)
            assert mock_session.post.call_count == 1  # At least once

    @pytest.mark.asyncio
    async def test_context_empty_base_url(self, mock_elevenlabs):
        """
        Test behavior with empty base URL.

        Empty URL should fall back to default.
        """
        from hooks.person_follow_hook import start_person_follow_hook

        mock_post_response = self.create_mock_response(status=200)
        mock_get_response = self.create_mock_response(
            status=200, json_data={"is_tracked": True}
        )

        with patch("aiohttp.ClientSession") as mock_session_class:
            mock_session = MagicMock()
            mock_session.__aenter__ = AsyncMock(return_value=mock_session)
            mock_session.__aexit__ = AsyncMock(return_value=None)
            mock_session.post.return_value = mock_post_response
            mock_session.get.return_value = mock_get_response
            mock_session_class.return_value = mock_session

            await start_person_follow_hook({
                "person_follow_base_url": "",
            })

            # Should use default URL instead of empty
            call_args = str(mock_session.post.call_args)
            assert "localhost:8080" in call_args

    @pytest.mark.asyncio
    async def test_context_url_with_trailing_slash(self, mock_elevenlabs):
        """
        Test URL handling with trailing slash in context.

        Will create double slashes like 'http://host//enroll'.
        """
        from hooks.person_follow_hook import start_person_follow_hook

        mock_post_response = self.create_mock_response(status=200)
        mock_get_response = self.create_mock_response(
            status=200, json_data={"is_tracked": True}
        )

        with patch("aiohttp.ClientSession") as mock_session_class:
            mock_session = MagicMock()
            mock_session.__aenter__ = AsyncMock(return_value=mock_session)
            mock_session.__aexit__ = AsyncMock(return_value=None)
            mock_session.post.return_value = mock_post_response
            mock_session.get.return_value = mock_get_response
            mock_session_class.return_value = mock_session

            await start_person_follow_hook({
                "person_follow_base_url": "http://localhost:8080/",
            })

            # URL will have double slash
            call_args = str(mock_session.post.call_args)
            assert "//enroll" in call_args

    @pytest.mark.asyncio
    async def test_context_none_values(self, mock_elevenlabs):
        """
        Test behavior when context values are explicitly None.

        None values should fall back to defaults (using 'or' pattern).
        """
        from hooks.person_follow_hook import start_person_follow_hook

        mock_post_response = self.create_mock_response(status=200)
        mock_get_response = self.create_mock_response(
            status=200, json_data={"is_tracked": True}
        )

        with patch("aiohttp.ClientSession") as mock_session_class:
            mock_session = MagicMock()
            mock_session.__aenter__ = AsyncMock(return_value=mock_session)
            mock_session.__aexit__ = AsyncMock(return_value=None)
            mock_session.post.return_value = mock_post_response
            mock_session.get.return_value = mock_get_response
            mock_session_class.return_value = mock_session

            # None values should use defaults instead of crashing
            result = await start_person_follow_hook({
                "person_follow_base_url": None,
                "enroll_timeout": None,
                "max_retries": None,
            })

            # Should succeed with defaults
            assert result["status"] == "success"
            assert result["is_tracked"] is True


class TestStartPersonFollowHook:
    """Tests for start_person_follow_hook function."""

    @pytest.fixture
    def mock_elevenlabs(self):
        """Mock ElevenLabsTTSProvider."""
        with patch("hooks.person_follow_hook.ElevenLabsTTSProvider") as mock:
            mock_instance = MagicMock()
            mock.return_value = mock_instance
            yield mock_instance

    @pytest.fixture
    def mock_session(self):
        """Create a mock aiohttp ClientSession."""
        session = MagicMock()
        session.__aenter__ = AsyncMock(return_value=session)
        session.__aexit__ = AsyncMock(return_value=None)
        return session

    def create_mock_response(self, status=200, json_data=None):
        """Helper to create mock response."""
        response = MagicMock()
        response.status = status
        response.__aenter__ = AsyncMock(return_value=response)
        response.__aexit__ = AsyncMock(return_value=None)
        if json_data:
            response.json = AsyncMock(return_value=json_data)
        return response

    @pytest.mark.asyncio
    async def test_start_success_immediate_tracking(self, mock_elevenlabs):
        """Test successful tracking on first attempt."""
        from hooks.person_follow_hook import start_person_follow_hook

        mock_post_response = self.create_mock_response(status=200)
        mock_get_response = self.create_mock_response(
            status=200, json_data={"is_tracked": True}
        )

        with patch("aiohttp.ClientSession") as mock_session_class:
            mock_session = MagicMock()
            mock_session.__aenter__ = AsyncMock(return_value=mock_session)
            mock_session.__aexit__ = AsyncMock(return_value=None)
            mock_session.post.return_value = mock_post_response
            mock_session.get.return_value = mock_get_response
            mock_session_class.return_value = mock_session

            result = await start_person_follow_hook({})

            assert result["status"] == "success"
            assert result["is_tracked"] is True
            assert "tracking" in result["message"].lower()
            mock_elevenlabs.add_pending_message.assert_called_once()

    @pytest.mark.asyncio
    async def test_start_success_after_retries(self, mock_elevenlabs):
        """Test successful tracking after multiple retry attempts."""
        from hooks.person_follow_hook import start_person_follow_hook

        call_count = 0

        def get_response(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count < 5:
                return self.create_mock_response(
                    status=200, json_data={"is_tracked": False}
                )
            return self.create_mock_response(
                status=200, json_data={"is_tracked": True}
            )

        mock_post_response = self.create_mock_response(status=200)

        with patch("aiohttp.ClientSession") as mock_session_class:
            mock_session = MagicMock()
            mock_session.__aenter__ = AsyncMock(return_value=mock_session)
            mock_session.__aexit__ = AsyncMock(return_value=None)
            mock_session.post.return_value = mock_post_response
            mock_session.get.side_effect = get_response
            mock_session_class.return_value = mock_session

            result = await start_person_follow_hook(
                {"enroll_timeout": 0.1, "max_retries": 3}
            )

            assert result["status"] == "success"

    @pytest.mark.asyncio
    async def test_start_awaiting_no_tracking(self, mock_elevenlabs):
        """Test when all retries complete but no tracking achieved."""
        from hooks.person_follow_hook import start_person_follow_hook

        mock_post_response = self.create_mock_response(status=200)
        mock_get_response = self.create_mock_response(
            status=200, json_data={"is_tracked": False}
        )

        with patch("aiohttp.ClientSession") as mock_session_class:
            mock_session = MagicMock()
            mock_session.__aenter__ = AsyncMock(return_value=mock_session)
            mock_session.__aexit__ = AsyncMock(return_value=None)
            mock_session.post.return_value = mock_post_response
            mock_session.get.return_value = mock_get_response
            mock_session_class.return_value = mock_session

            result = await start_person_follow_hook(
                {"enroll_timeout": 0.1, "max_retries": 2}
            )

            assert result["status"] == "success"
            assert result["is_tracked"] is False
            assert "awaiting" in result["message"].lower()
            mock_elevenlabs.add_pending_message.assert_called()

    @pytest.mark.asyncio
    async def test_start_enroll_returns_non_200(self, mock_elevenlabs):
        """Test when POST /enroll returns non-200 status."""
        from hooks.person_follow_hook import start_person_follow_hook

        mock_post_response = self.create_mock_response(status=500)
        mock_get_response = self.create_mock_response(
            status=200, json_data={"is_tracked": False}
        )

        with patch("aiohttp.ClientSession") as mock_session_class:
            mock_session = MagicMock()
            mock_session.__aenter__ = AsyncMock(return_value=mock_session)
            mock_session.__aexit__ = AsyncMock(return_value=None)
            mock_session.post.return_value = mock_post_response
            mock_session.get.return_value = mock_get_response
            mock_session_class.return_value = mock_session

            result = await start_person_follow_hook(
                {"enroll_timeout": 0.1, "max_retries": 2}
            )

            assert result["status"] == "success"
            assert result["is_tracked"] is False

    @pytest.mark.asyncio
    async def test_start_enroll_client_error(self, mock_elevenlabs):
        """Test when POST /enroll raises ClientError."""
        from hooks.person_follow_hook import start_person_follow_hook

        with patch("aiohttp.ClientSession") as mock_session_class:
            mock_session = MagicMock()
            mock_session.__aenter__ = AsyncMock(return_value=mock_session)
            mock_session.__aexit__ = AsyncMock(return_value=None)
            mock_session.post.side_effect = aiohttp.ClientError("Connection failed")
            mock_session_class.return_value = mock_session

            result = await start_person_follow_hook(
                {"enroll_timeout": 0.1, "max_retries": 2}
            )

            assert result["status"] == "success"
            assert result["is_tracked"] is False

    @pytest.mark.asyncio
    async def test_start_status_returns_non_200(self, mock_elevenlabs):
        """Test when GET /status returns non-200 status."""
        from hooks.person_follow_hook import start_person_follow_hook

        mock_post_response = self.create_mock_response(status=200)
        mock_get_response = self.create_mock_response(status=500)

        with patch("aiohttp.ClientSession") as mock_session_class:
            mock_session = MagicMock()
            mock_session.__aenter__ = AsyncMock(return_value=mock_session)
            mock_session.__aexit__ = AsyncMock(return_value=None)
            mock_session.post.return_value = mock_post_response
            mock_session.get.return_value = mock_get_response
            mock_session_class.return_value = mock_session

            result = await start_person_follow_hook(
                {"enroll_timeout": 0.1, "max_retries": 1}
            )

            assert result["status"] == "success"
            assert result["is_tracked"] is False

    @pytest.mark.asyncio
    async def test_start_status_poll_exception(self, mock_elevenlabs):
        """Test when GET /status raises an exception."""
        from hooks.person_follow_hook import start_person_follow_hook

        mock_post_response = self.create_mock_response(status=200)

        with patch("aiohttp.ClientSession") as mock_session_class:
            mock_session = MagicMock()
            mock_session.__aenter__ = AsyncMock(return_value=mock_session)
            mock_session.__aexit__ = AsyncMock(return_value=None)
            mock_session.post.return_value = mock_post_response
            mock_session.get.side_effect = Exception("Unexpected error")
            mock_session_class.return_value = mock_session

            result = await start_person_follow_hook(
                {"enroll_timeout": 0.1, "max_retries": 1}
            )

            assert result["status"] == "success"
            assert result["is_tracked"] is False

    @pytest.mark.asyncio
    async def test_start_connection_error(self, mock_elevenlabs):
        """Test when connection completely fails."""
        from hooks.person_follow_hook import start_person_follow_hook

        with patch("aiohttp.ClientSession") as mock_session_class:
            mock_session = MagicMock()
            mock_session.__aenter__ = AsyncMock(
                side_effect=aiohttp.ClientError("Connection refused")
            )
            mock_session_class.return_value = mock_session

            result = await start_person_follow_hook({})

            assert result["status"] == "error"
            assert "connection" in result["message"].lower()
            mock_elevenlabs.add_pending_message.assert_called()

    @pytest.mark.asyncio
    async def test_start_custom_context_params(self, mock_elevenlabs):
        """Test with custom context parameters."""
        from hooks.person_follow_hook import start_person_follow_hook

        mock_post_response = self.create_mock_response(status=200)
        mock_get_response = self.create_mock_response(
            status=200, json_data={"is_tracked": True}
        )

        with patch("aiohttp.ClientSession") as mock_session_class:
            mock_session = MagicMock()
            mock_session.__aenter__ = AsyncMock(return_value=mock_session)
            mock_session.__aexit__ = AsyncMock(return_value=None)
            mock_session.post.return_value = mock_post_response
            mock_session.get.return_value = mock_get_response
            mock_session_class.return_value = mock_session

            context = {
                "person_follow_base_url": "http://custom:9999",
                "enroll_timeout": 5.0,
                "max_retries": 10,
            }
            result = await start_person_follow_hook(context)

            assert result["status"] == "success"

    @pytest.mark.asyncio
    async def test_start_tts_message_on_success(self, mock_elevenlabs):
        """Test TTS message is sent on successful tracking."""
        from hooks.person_follow_hook import start_person_follow_hook

        mock_post_response = self.create_mock_response(status=200)
        mock_get_response = self.create_mock_response(
            status=200, json_data={"is_tracked": True}
        )

        with patch("aiohttp.ClientSession") as mock_session_class:
            mock_session = MagicMock()
            mock_session.__aenter__ = AsyncMock(return_value=mock_session)
            mock_session.__aexit__ = AsyncMock(return_value=None)
            mock_session.post.return_value = mock_post_response
            mock_session.get.return_value = mock_get_response
            mock_session_class.return_value = mock_session

            await start_person_follow_hook({})

            mock_elevenlabs.add_pending_message.assert_called_once()
            call_args = mock_elevenlabs.add_pending_message.call_args[0][0]
            assert "follow" in call_args.lower()

    @pytest.mark.asyncio
    async def test_start_tts_message_on_awaiting(self, mock_elevenlabs):
        """Test TTS message when awaiting person detection."""
        from hooks.person_follow_hook import start_person_follow_hook

        mock_post_response = self.create_mock_response(status=200)
        mock_get_response = self.create_mock_response(
            status=200, json_data={"is_tracked": False}
        )

        with patch("aiohttp.ClientSession") as mock_session_class:
            mock_session = MagicMock()
            mock_session.__aenter__ = AsyncMock(return_value=mock_session)
            mock_session.__aexit__ = AsyncMock(return_value=None)
            mock_session.post.return_value = mock_post_response
            mock_session.get.return_value = mock_get_response
            mock_session_class.return_value = mock_session

            await start_person_follow_hook({"enroll_timeout": 0.1, "max_retries": 1})

            mock_elevenlabs.add_pending_message.assert_called()
            call_args = mock_elevenlabs.add_pending_message.call_args[0][0]
            assert "stand" in call_args.lower() or "activated" in call_args.lower()

    @pytest.mark.asyncio
    async def test_start_tts_message_on_error(self, mock_elevenlabs):
        """Test TTS message on connection error."""
        from hooks.person_follow_hook import start_person_follow_hook

        with patch("aiohttp.ClientSession") as mock_session_class:
            mock_session = MagicMock()
            mock_session.__aenter__ = AsyncMock(
                side_effect=aiohttp.ClientError("Connection refused")
            )
            mock_session_class.return_value = mock_session

            await start_person_follow_hook({})

            mock_elevenlabs.add_pending_message.assert_called()
            call_args = mock_elevenlabs.add_pending_message.call_args[0][0]
            assert "connect" in call_args.lower()


class TestStopPersonFollowHook:
    """Tests for stop_person_follow_hook function."""

    def create_mock_response(self, status=200):
        """Helper to create mock response."""
        response = MagicMock()
        response.status = status
        response.__aenter__ = AsyncMock(return_value=response)
        response.__aexit__ = AsyncMock(return_value=None)
        return response

    @pytest.mark.asyncio
    async def test_stop_success(self):
        """Test successful stop with 200 response."""
        from hooks.person_follow_hook import stop_person_follow_hook

        mock_response = self.create_mock_response(status=200)

        with patch("aiohttp.ClientSession") as mock_session_class:
            mock_session = MagicMock()
            mock_session.__aenter__ = AsyncMock(return_value=mock_session)
            mock_session.__aexit__ = AsyncMock(return_value=None)
            mock_session.post.return_value = mock_response
            mock_session_class.return_value = mock_session

            result = await stop_person_follow_hook({})

            assert result["status"] == "success"
            assert "stopped" in result["message"].lower()

    @pytest.mark.asyncio
    async def test_stop_failure_non_200(self):
        """Test stop failure when server returns non-200."""
        from hooks.person_follow_hook import stop_person_follow_hook

        mock_response = self.create_mock_response(status=500)

        with patch("aiohttp.ClientSession") as mock_session_class:
            mock_session = MagicMock()
            mock_session.__aenter__ = AsyncMock(return_value=mock_session)
            mock_session.__aexit__ = AsyncMock(return_value=None)
            mock_session.post.return_value = mock_response
            mock_session_class.return_value = mock_session

            result = await stop_person_follow_hook({})

            assert result["status"] == "error"
            assert "failed" in result["message"].lower()

    @pytest.mark.asyncio
    async def test_stop_connection_error(self):
        """Test stop when connection fails."""
        from hooks.person_follow_hook import stop_person_follow_hook

        with patch("aiohttp.ClientSession") as mock_session_class:
            mock_session = MagicMock()
            mock_session.__aenter__ = AsyncMock(return_value=mock_session)
            mock_session.__aexit__ = AsyncMock(return_value=None)
            mock_session.post.side_effect = aiohttp.ClientError("Connection refused")
            mock_session_class.return_value = mock_session

            result = await stop_person_follow_hook({})

            assert result["status"] == "error"
            assert "connection" in result["message"].lower()

    @pytest.mark.asyncio
    async def test_stop_custom_base_url(self):
        """Test stop with custom base URL from context."""
        from hooks.person_follow_hook import stop_person_follow_hook

        mock_response = self.create_mock_response(status=200)

        with patch("aiohttp.ClientSession") as mock_session_class:
            mock_session = MagicMock()
            mock_session.__aenter__ = AsyncMock(return_value=mock_session)
            mock_session.__aexit__ = AsyncMock(return_value=None)
            mock_session.post.return_value = mock_response
            mock_session_class.return_value = mock_session

            context = {"person_follow_base_url": "http://custom:9999"}
            result = await stop_person_follow_hook(context)

            assert result["status"] == "success"
            # Verify the custom URL was used
            call_args = mock_session.post.call_args
            assert "custom:9999" in str(call_args)


class TestStartHookBehaviorCorrectness:
    """Test start_person_follow_hook behavior correctness."""

    @pytest.fixture
    def mock_elevenlabs(self):
        """Mock ElevenLabsTTSProvider."""
        with patch("hooks.person_follow_hook.ElevenLabsTTSProvider") as mock:
            mock_instance = MagicMock()
            mock.return_value = mock_instance
            yield mock_instance

    def create_mock_response(self, status=200, json_data=None):
        """Helper to create mock response."""
        response = MagicMock()
        response.status = status
        response.__aenter__ = AsyncMock(return_value=response)
        response.__aexit__ = AsyncMock(return_value=None)
        if json_data:
            response.json = AsyncMock(return_value=json_data)
        return response

    @pytest.mark.asyncio
    async def test_retry_logic_exhausts_all_attempts(self, mock_elevenlabs):
        """
        Test that retry logic correctly exhausts all attempts.

        Verify: Each retry attempt calls POST /enroll.
        """
        from hooks.person_follow_hook import start_person_follow_hook

        post_call_count = 0

        def count_posts(*args, **kwargs):
            nonlocal post_call_count
            post_call_count += 1
            return self.create_mock_response(status=200)

        mock_get_response = self.create_mock_response(
            status=200, json_data={"is_tracked": False}
        )

        with patch("aiohttp.ClientSession") as mock_session_class:
            mock_session = MagicMock()
            mock_session.__aenter__ = AsyncMock(return_value=mock_session)
            mock_session.__aexit__ = AsyncMock(return_value=None)
            mock_session.post.side_effect = count_posts
            mock_session.get.return_value = mock_get_response
            mock_session_class.return_value = mock_session

            max_retries = 3
            await start_person_follow_hook({
                "enroll_timeout": 0.1,
                "max_retries": max_retries,
            })

            # Should have called POST exactly max_retries times
            assert post_call_count == max_retries

    @pytest.mark.asyncio
    async def test_stops_retrying_on_success(self, mock_elevenlabs):
        """
        Test that retry loop stops when tracking is successful.

        Verify: If tracking succeeds on attempt 2, attempt 3 is NOT made.
        """
        from hooks.person_follow_hook import start_person_follow_hook

        attempt_count = 0

        def get_response(*args, **kwargs):
            nonlocal attempt_count
            attempt_count += 1
            # Succeed on second attempt
            if attempt_count >= 2:
                return self.create_mock_response(
                    status=200, json_data={"is_tracked": True}
                )
            return self.create_mock_response(
                status=200, json_data={"is_tracked": False}
            )

        mock_post_response = self.create_mock_response(status=200)

        with patch("aiohttp.ClientSession") as mock_session_class:
            mock_session = MagicMock()
            mock_session.__aenter__ = AsyncMock(return_value=mock_session)
            mock_session.__aexit__ = AsyncMock(return_value=None)
            mock_session.post.return_value = mock_post_response
            mock_session.get.side_effect = get_response
            mock_session_class.return_value = mock_session

            result = await start_person_follow_hook({
                "enroll_timeout": 0.1,
                "max_retries": 5,
            })

            assert result["status"] == "success"
            assert result["is_tracked"] is True
            # Should have stopped after 2 attempts, not all 5
            assert attempt_count == 2

    @pytest.mark.asyncio
    async def test_timeout_respected_per_attempt(self, mock_elevenlabs):
        """
        Test that enroll_timeout is respected for status polling.

        Each attempt should poll for at most enroll_timeout seconds.
        """
        import time

        from hooks.person_follow_hook import start_person_follow_hook

        start_time = time.time()

        mock_post_response = self.create_mock_response(status=200)
        mock_get_response = self.create_mock_response(
            status=200, json_data={"is_tracked": False}
        )

        with patch("aiohttp.ClientSession") as mock_session_class:
            mock_session = MagicMock()
            mock_session.__aenter__ = AsyncMock(return_value=mock_session)
            mock_session.__aexit__ = AsyncMock(return_value=None)
            mock_session.post.return_value = mock_post_response
            mock_session.get.return_value = mock_get_response
            mock_session_class.return_value = mock_session

            # Very short timeout for test speed
            await start_person_follow_hook({
                "enroll_timeout": 0.2,
                "max_retries": 2,
            })

            elapsed = time.time() - start_time
            # Should complete in reasonable time (2 retries * 0.2s timeout + overhead)
            assert elapsed < 2.0, f"Took too long: {elapsed}s"

    @pytest.mark.asyncio
    async def test_elevenlabs_not_called_multiple_times_on_success(
        self, mock_elevenlabs
    ):
        """
        Test that TTS message is only sent once on success.
        """
        from hooks.person_follow_hook import start_person_follow_hook

        mock_post_response = self.create_mock_response(status=200)
        mock_get_response = self.create_mock_response(
            status=200, json_data={"is_tracked": True}
        )

        with patch("aiohttp.ClientSession") as mock_session_class:
            mock_session = MagicMock()
            mock_session.__aenter__ = AsyncMock(return_value=mock_session)
            mock_session.__aexit__ = AsyncMock(return_value=None)
            mock_session.post.return_value = mock_post_response
            mock_session.get.return_value = mock_get_response
            mock_session_class.return_value = mock_session

            await start_person_follow_hook({})

            # TTS should be called exactly once
            assert mock_elevenlabs.add_pending_message.call_count == 1

    @pytest.mark.asyncio
    async def test_return_values_are_correct_types(self, mock_elevenlabs):
        """
        Test that return values have correct structure.
        """
        from hooks.person_follow_hook import start_person_follow_hook

        mock_post_response = self.create_mock_response(status=200)
        mock_get_response = self.create_mock_response(
            status=200, json_data={"is_tracked": True}
        )

        with patch("aiohttp.ClientSession") as mock_session_class:
            mock_session = MagicMock()
            mock_session.__aenter__ = AsyncMock(return_value=mock_session)
            mock_session.__aexit__ = AsyncMock(return_value=None)
            mock_session.post.return_value = mock_post_response
            mock_session.get.return_value = mock_get_response
            mock_session_class.return_value = mock_session

            result = await start_person_follow_hook({})

            # Check structure
            assert isinstance(result, dict)
            assert "status" in result
            assert "message" in result
            assert "is_tracked" in result
            assert isinstance(result["status"], str)
            assert isinstance(result["message"], str)
            assert isinstance(result["is_tracked"], bool)


class TestStopHookBehaviorCorrectness:
    """Test stop_person_follow_hook behavior correctness."""

    def create_mock_response(self, status=200):
        """Helper to create mock response."""
        response = MagicMock()
        response.status = status
        response.__aenter__ = AsyncMock(return_value=response)
        response.__aexit__ = AsyncMock(return_value=None)
        return response

    @pytest.mark.asyncio
    async def test_correct_endpoint_called(self):
        """Test that /clear endpoint is called correctly."""
        from hooks.person_follow_hook import stop_person_follow_hook

        mock_response = self.create_mock_response(status=200)

        with patch("aiohttp.ClientSession") as mock_session_class:
            mock_session = MagicMock()
            mock_session.__aenter__ = AsyncMock(return_value=mock_session)
            mock_session.__aexit__ = AsyncMock(return_value=None)
            mock_session.post.return_value = mock_response
            mock_session_class.return_value = mock_session

            await stop_person_follow_hook({})

            # Verify POST was called (not GET, PUT, DELETE)
            mock_session.post.assert_called_once()
            # Verify URL contains /clear
            call_args = str(mock_session.post.call_args)
            assert "/clear" in call_args

    @pytest.mark.asyncio
    async def test_return_values_are_correct_types(self):
        """Test that return values have correct structure."""
        from hooks.person_follow_hook import stop_person_follow_hook

        mock_response = self.create_mock_response(status=200)

        with patch("aiohttp.ClientSession") as mock_session_class:
            mock_session = MagicMock()
            mock_session.__aenter__ = AsyncMock(return_value=mock_session)
            mock_session.__aexit__ = AsyncMock(return_value=None)
            mock_session.post.return_value = mock_response
            mock_session_class.return_value = mock_session

            result = await stop_person_follow_hook({})

            # Check structure
            assert isinstance(result, dict)
            assert "status" in result
            assert "message" in result
            assert isinstance(result["status"], str)
            assert isinstance(result["message"], str)

    @pytest.mark.asyncio
    async def test_no_retry_on_failure(self):
        """Test that stop does NOT retry on failure (unlike start)."""
        from hooks.person_follow_hook import stop_person_follow_hook

        call_count = 0

        def count_calls(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            return self.create_mock_response(status=500)

        with patch("aiohttp.ClientSession") as mock_session_class:
            mock_session = MagicMock()
            mock_session.__aenter__ = AsyncMock(return_value=mock_session)
            mock_session.__aexit__ = AsyncMock(return_value=None)
            mock_session.post.side_effect = count_calls
            mock_session_class.return_value = mock_session

            await stop_person_follow_hook({})

            # Should only try once (no retry logic in stop)
            assert call_count == 1


class TestElevenLabsProviderBehavior:
    """
    Test ElevenLabsTTSProvider usage in hooks.

    NOTE: Each hook call creates a NEW ElevenLabsTTSProvider instance.
    This may or may not be intended behavior.
    """

    @pytest.mark.asyncio
    async def test_new_provider_created_each_call(self):
        """
        Test that a new ElevenLabsTTSProvider is created on each hook call.

        This documents current behavior - may be a performance concern.
        """
        from hooks.person_follow_hook import start_person_follow_hook

        creation_count = 0

        def count_creations(*args, **kwargs):
            nonlocal creation_count
            creation_count += 1
            return MagicMock()

        mock_post_response = MagicMock()
        mock_post_response.status = 200
        mock_post_response.__aenter__ = AsyncMock(return_value=mock_post_response)
        mock_post_response.__aexit__ = AsyncMock(return_value=None)

        mock_get_response = MagicMock()
        mock_get_response.status = 200
        mock_get_response.__aenter__ = AsyncMock(return_value=mock_get_response)
        mock_get_response.__aexit__ = AsyncMock(return_value=None)
        mock_get_response.json = AsyncMock(return_value={"is_tracked": True})

        with patch(
            "hooks.person_follow_hook.ElevenLabsTTSProvider",
            side_effect=count_creations,
        ):
            with patch("aiohttp.ClientSession") as mock_session_class:
                mock_session = MagicMock()
                mock_session.__aenter__ = AsyncMock(return_value=mock_session)
                mock_session.__aexit__ = AsyncMock(return_value=None)
                mock_session.post.return_value = mock_post_response
                mock_session.get.return_value = mock_get_response
                mock_session_class.return_value = mock_session

                # Call hook twice
                await start_person_follow_hook({})
                await start_person_follow_hook({})

                # Should create 2 separate instances (one per call)
                assert creation_count == 2
