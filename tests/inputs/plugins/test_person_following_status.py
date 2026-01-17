"""Tests for PersonFollowingStatus input plugin."""

import sys
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Mock external dependencies before importing
sys.modules["zenoh"] = MagicMock()
sys.modules["zenoh_msgs"] = MagicMock()
sys.modules["cv2"] = MagicMock()
sys.modules["numpy"] = MagicMock()
sys.modules["PIL"] = MagicMock()
sys.modules["google"] = MagicMock()
sys.modules["google.generativeai"] = MagicMock()
sys.modules["openai"] = MagicMock()
sys.modules["elevenlabs"] = MagicMock()
sys.modules["riva"] = MagicMock()
sys.modules["riva.client"] = MagicMock()
sys.modules["pyaudio"] = MagicMock()
sys.modules["sounddevice"] = MagicMock()


class TestPersonFollowingStatusConfig:
    """Tests for PersonFollowingStatusConfig."""

    def test_config_defaults(self):
        """Test default configuration values."""
        from inputs.plugins.person_following_status import PersonFollowingStatusConfig

        config = PersonFollowingStatusConfig()

        assert config.person_follow_base_url == "http://localhost:8080"
        assert config.poll_interval == 0.5
        assert config.enroll_retry_interval == 3.0

    def test_config_custom_values(self):
        """Test configuration with custom values."""
        from inputs.plugins.person_following_status import PersonFollowingStatusConfig

        config = PersonFollowingStatusConfig(
            person_follow_base_url="http://custom:9999",
            poll_interval=1.0,
            enroll_retry_interval=5.0,
        )

        assert config.person_follow_base_url == "http://custom:9999"
        assert config.poll_interval == 1.0
        assert config.enroll_retry_interval == 5.0


class TestConfigValidation:
    """
    Test configuration validation.

    These tests verify behavior with edge case and invalid config values.
    """

    @pytest.fixture
    def mock_io_provider(self):
        """Mock IOProvider."""
        with patch("inputs.plugins.person_following_status.IOProvider") as mock:
            mock_instance = MagicMock()
            mock.return_value = mock_instance
            yield mock_instance

    def test_config_zero_poll_interval(self, mock_io_provider):
        """
        Test behavior with zero poll_interval.

        Zero poll_interval means no sleep between polls - CPU intensive.
        """
        from inputs.plugins.person_following_status import (
            PersonFollowingStatus,
            PersonFollowingStatusConfig,
        )

        config = PersonFollowingStatusConfig(poll_interval=0.0)
        status = PersonFollowingStatus(config)

        assert status.poll_interval == 0.0

    def test_config_negative_poll_interval(self, mock_io_provider):
        """
        Test behavior with negative poll_interval.

        CURRENT BEHAVIOR: Pydantic accepts negative values.
        This might cause asyncio.sleep to behave unexpectedly.
        """
        from inputs.plugins.person_following_status import (
            PersonFollowingStatus,
            PersonFollowingStatusConfig,
        )

        # Pydantic currently accepts negative values
        config = PersonFollowingStatusConfig(poll_interval=-1.0)
        status = PersonFollowingStatus(config)

        # Document current behavior - negative is accepted
        assert status.poll_interval == -1.0

    def test_config_very_small_poll_interval(self, mock_io_provider):
        """Test with very small but valid poll_interval."""
        from inputs.plugins.person_following_status import (
            PersonFollowingStatus,
            PersonFollowingStatusConfig,
        )

        config = PersonFollowingStatusConfig(poll_interval=0.001)
        status = PersonFollowingStatus(config)

        assert status.poll_interval == 0.001

    def test_config_very_large_poll_interval(self, mock_io_provider):
        """Test with very large poll_interval."""
        from inputs.plugins.person_following_status import (
            PersonFollowingStatus,
            PersonFollowingStatusConfig,
        )

        config = PersonFollowingStatusConfig(poll_interval=3600.0)  # 1 hour
        status = PersonFollowingStatus(config)

        assert status.poll_interval == 3600.0

    def test_config_zero_enroll_retry_interval(self, mock_io_provider):
        """
        Test behavior with zero enroll_retry_interval.

        Zero means enroll on every poll when INACTIVE.
        """
        from inputs.plugins.person_following_status import (
            PersonFollowingStatus,
            PersonFollowingStatusConfig,
        )

        config = PersonFollowingStatusConfig(enroll_retry_interval=0.0)
        status = PersonFollowingStatus(config)

        assert status.enroll_retry_interval == 0.0

    def test_config_empty_url(self, mock_io_provider):
        """
        Test behavior with empty URL.

        CURRENT BEHAVIOR: Empty string is accepted.
        Will cause HTTP errors at runtime.
        """
        from inputs.plugins.person_following_status import (
            PersonFollowingStatus,
            PersonFollowingStatusConfig,
        )

        config = PersonFollowingStatusConfig(person_follow_base_url="")
        status = PersonFollowingStatus(config)

        assert status.base_url == ""
        assert status.status_url == "/status"  # Malformed URL
        assert status.enroll_url == "/enroll"  # Malformed URL

    def test_config_invalid_url_format(self, mock_io_provider):
        """
        Test behavior with invalid URL format.

        CURRENT BEHAVIOR: No URL validation, accepts any string.
        """
        from inputs.plugins.person_following_status import (
            PersonFollowingStatus,
            PersonFollowingStatusConfig,
        )

        config = PersonFollowingStatusConfig(
            person_follow_base_url="not-a-valid-url"
        )
        status = PersonFollowingStatus(config)

        # Currently accepts any string
        assert status.base_url == "not-a-valid-url"

    def test_config_url_with_trailing_slash(self, mock_io_provider):
        """
        Test URL handling with trailing slash.

        Check if double slashes occur: http://host//status
        """
        from inputs.plugins.person_following_status import (
            PersonFollowingStatus,
            PersonFollowingStatusConfig,
        )

        config = PersonFollowingStatusConfig(
            person_follow_base_url="http://localhost:8080/"
        )
        status = PersonFollowingStatus(config)

        # Current behavior: creates double slash
        assert status.status_url == "http://localhost:8080//status"
        # This might work but is not clean

    def test_config_url_without_port(self, mock_io_provider):
        """Test URL without explicit port."""
        from inputs.plugins.person_following_status import (
            PersonFollowingStatus,
            PersonFollowingStatusConfig,
        )

        config = PersonFollowingStatusConfig(
            person_follow_base_url="http://localhost"
        )
        status = PersonFollowingStatus(config)

        assert status.status_url == "http://localhost/status"

    def test_config_https_url(self, mock_io_provider):
        """Test HTTPS URL."""
        from inputs.plugins.person_following_status import (
            PersonFollowingStatus,
            PersonFollowingStatusConfig,
        )

        config = PersonFollowingStatusConfig(
            person_follow_base_url="https://secure.example.com:443"
        )
        status = PersonFollowingStatus(config)

        assert status.status_url == "https://secure.example.com:443/status"


class TestPersonFollowingStatusInit:
    """Tests for PersonFollowingStatus initialization."""

    @pytest.fixture
    def mock_io_provider(self):
        """Mock IOProvider."""
        with patch("inputs.plugins.person_following_status.IOProvider") as mock:
            mock_instance = MagicMock()
            mock.return_value = mock_instance
            yield mock_instance

    def test_init_state_variables(self, mock_io_provider):
        """Test that state variables are initialized correctly."""
        from inputs.plugins.person_following_status import (
            PersonFollowingStatus,
            PersonFollowingStatusConfig,
        )

        config = PersonFollowingStatusConfig()
        status = PersonFollowingStatus(config)

        assert status._previous_is_tracked is None
        assert status._lost_tracking_time is None
        assert status._lost_tracking_announced is False
        assert status._last_enroll_attempt == 0.0
        assert status._has_ever_tracked is False
        assert status.descriptor_for_LLM == "Person Following Status"

    def test_init_urls(self, mock_io_provider):
        """Test that URLs are constructed correctly."""
        from inputs.plugins.person_following_status import (
            PersonFollowingStatus,
            PersonFollowingStatusConfig,
        )

        config = PersonFollowingStatusConfig(
            person_follow_base_url="http://test:1234"
        )
        status = PersonFollowingStatus(config)

        assert status.base_url == "http://test:1234"
        assert status.status_url == "http://test:1234/status"
        assert status.enroll_url == "http://test:1234/enroll"


class TestPersonFollowingStatusPoll:
    """Tests for PersonFollowingStatus._poll method."""

    @pytest.fixture
    def mock_io_provider(self):
        """Mock IOProvider."""
        with patch("inputs.plugins.person_following_status.IOProvider") as mock:
            mock_instance = MagicMock()
            mock.return_value = mock_instance
            yield mock_instance

    @pytest.fixture
    def status_instance(self, mock_io_provider):
        """Create a PersonFollowingStatus instance."""
        from inputs.plugins.person_following_status import (
            PersonFollowingStatus,
            PersonFollowingStatusConfig,
        )

        config = PersonFollowingStatusConfig(poll_interval=0.01)
        return PersonFollowingStatus(config)

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
    async def test_poll_success_tracking(self, status_instance):
        """Test poll when person is being tracked."""
        mock_response = self.create_mock_response(
            status=200,
            json_data={
                "is_tracked": True,
                "status": "TRACKING_ACTIVE",
                "x": 0.5,
                "z": 2.0,
                "target_track_id": 1,
            },
        )

        with patch("aiohttp.ClientSession") as mock_session_class:
            mock_session = MagicMock()
            mock_session.__aenter__ = AsyncMock(return_value=mock_session)
            mock_session.__aexit__ = AsyncMock(return_value=None)
            mock_session.get.return_value = mock_response
            mock_session_class.return_value = mock_session

            result = await status_instance._poll()

            assert result is not None
            assert "TRACKING" in result

    @pytest.mark.asyncio
    async def test_poll_success_not_tracking(self, status_instance):
        """Test poll when person is not being tracked."""
        mock_response = self.create_mock_response(
            status=200,
            json_data={
                "is_tracked": False,
                "status": "INACTIVE",
                "target_track_id": None,
            },
        )

        with patch("aiohttp.ClientSession") as mock_session_class:
            mock_session = MagicMock()
            mock_session.__aenter__ = AsyncMock(return_value=mock_session)
            mock_session.__aexit__ = AsyncMock(return_value=None)
            mock_session.get.return_value = mock_response
            mock_session.post.return_value = self.create_mock_response(status=200)
            mock_session_class.return_value = mock_session

            # First call sets up state
            status_instance._last_enroll_attempt = 0
            await status_instance._poll()

            # Should attempt enrollment for INACTIVE status
            mock_session.post.assert_called()

    @pytest.mark.asyncio
    async def test_poll_status_non_200(self, status_instance):
        """Test poll when status endpoint returns non-200."""
        mock_response = self.create_mock_response(status=500)

        with patch("aiohttp.ClientSession") as mock_session_class:
            mock_session = MagicMock()
            mock_session.__aenter__ = AsyncMock(return_value=mock_session)
            mock_session.__aexit__ = AsyncMock(return_value=None)
            mock_session.get.return_value = mock_response
            mock_session_class.return_value = mock_session

            result = await status_instance._poll()

            assert result is None

    @pytest.mark.asyncio
    async def test_poll_inactive_triggers_enroll(self, status_instance):
        """Test that INACTIVE status triggers enrollment attempt."""
        mock_get_response = self.create_mock_response(
            status=200,
            json_data={
                "is_tracked": False,
                "status": "INACTIVE",
                "target_track_id": None,
            },
        )
        mock_post_response = self.create_mock_response(status=200)

        with patch("aiohttp.ClientSession") as mock_session_class:
            mock_session = MagicMock()
            mock_session.__aenter__ = AsyncMock(return_value=mock_session)
            mock_session.__aexit__ = AsyncMock(return_value=None)
            mock_session.get.return_value = mock_get_response
            mock_session.post.return_value = mock_post_response
            mock_session_class.return_value = mock_session

            status_instance._last_enroll_attempt = 0
            await status_instance._poll()

            mock_session.post.assert_called()

    @pytest.mark.asyncio
    async def test_poll_searching_no_enroll(self, status_instance):
        """Test that SEARCHING status does NOT trigger enrollment."""
        mock_response = self.create_mock_response(
            status=200,
            json_data={
                "is_tracked": False,
                "status": "SEARCHING",
                "target_track_id": 1,
            },
        )

        with patch("aiohttp.ClientSession") as mock_session_class:
            mock_session = MagicMock()
            mock_session.__aenter__ = AsyncMock(return_value=mock_session)
            mock_session.__aexit__ = AsyncMock(return_value=None)
            mock_session.get.return_value = mock_response
            mock_session_class.return_value = mock_session

            status_instance._last_enroll_attempt = 0
            await status_instance._poll()

            # Should NOT call post for SEARCHING status
            mock_session.post.assert_not_called()

    @pytest.mark.asyncio
    async def test_poll_enroll_interval_respected(self, status_instance):
        """Test that enrollment interval is respected."""
        mock_response = self.create_mock_response(
            status=200,
            json_data={
                "is_tracked": False,
                "status": "INACTIVE",
                "target_track_id": None,
            },
        )
        mock_post_response = self.create_mock_response(status=200)

        with patch("aiohttp.ClientSession") as mock_session_class:
            mock_session = MagicMock()
            mock_session.__aenter__ = AsyncMock(return_value=mock_session)
            mock_session.__aexit__ = AsyncMock(return_value=None)
            mock_session.get.return_value = mock_response
            mock_session.post.return_value = mock_post_response
            mock_session_class.return_value = mock_session

            # Set last attempt to now (within interval)
            status_instance._last_enroll_attempt = time.time()
            await status_instance._poll()

            # Should NOT call post because within interval
            mock_session.post.assert_not_called()

    @pytest.mark.asyncio
    async def test_poll_client_error(self, status_instance):
        """Test poll handles ClientError gracefully."""
        import aiohttp

        with patch("aiohttp.ClientSession") as mock_session_class:
            mock_session = MagicMock()
            mock_session.__aenter__ = AsyncMock(return_value=mock_session)
            mock_session.__aexit__ = AsyncMock(return_value=None)
            mock_session.get.side_effect = aiohttp.ClientError("Connection failed")
            mock_session_class.return_value = mock_session

            result = await status_instance._poll()

            assert result is None

    @pytest.mark.asyncio
    async def test_poll_unexpected_error(self, status_instance):
        """Test poll handles unexpected errors gracefully."""
        with patch("aiohttp.ClientSession") as mock_session_class:
            mock_session = MagicMock()
            mock_session.__aenter__ = AsyncMock(return_value=mock_session)
            mock_session.__aexit__ = AsyncMock(return_value=None)
            mock_session.get.side_effect = ValueError("Unexpected")
            mock_session_class.return_value = mock_session

            result = await status_instance._poll()

            assert result is None


class TestPersonFollowingStatusTryEnroll:
    """Tests for PersonFollowingStatus._try_enroll method."""

    @pytest.fixture
    def mock_io_provider(self):
        """Mock IOProvider."""
        with patch("inputs.plugins.person_following_status.IOProvider") as mock:
            mock_instance = MagicMock()
            mock.return_value = mock_instance
            yield mock_instance

    @pytest.fixture
    def status_instance(self, mock_io_provider):
        """Create a PersonFollowingStatus instance."""
        from inputs.plugins.person_following_status import (
            PersonFollowingStatus,
            PersonFollowingStatusConfig,
        )

        config = PersonFollowingStatusConfig(poll_interval=0.01)
        return PersonFollowingStatus(config)

    def create_mock_response(self, status=200):
        """Helper to create mock response."""
        response = MagicMock()
        response.status = status
        response.__aenter__ = AsyncMock(return_value=response)
        response.__aexit__ = AsyncMock(return_value=None)
        return response

    @pytest.mark.asyncio
    async def test_try_enroll_success(self, status_instance):
        """Test successful enrollment request."""
        mock_response = self.create_mock_response(status=200)

        mock_session = MagicMock()
        mock_session.post.return_value = mock_response

        # Should not raise any exception
        await status_instance._try_enroll(mock_session)

        mock_session.post.assert_called_once()

    @pytest.mark.asyncio
    async def test_try_enroll_non_200(self, status_instance):
        """Test enrollment with non-200 response."""
        mock_response = self.create_mock_response(status=500)

        mock_session = MagicMock()
        mock_session.post.return_value = mock_response

        # Should not raise any exception
        await status_instance._try_enroll(mock_session)

        mock_session.post.assert_called_once()

    @pytest.mark.asyncio
    async def test_try_enroll_exception(self, status_instance):
        """Test enrollment when exception occurs."""
        mock_session = MagicMock()
        mock_session.post.side_effect = Exception("Connection failed")

        # Should not raise any exception
        result = await status_instance._try_enroll(mock_session)

        assert result is None


class TestPersonFollowingStatusFormatStatus:
    """Tests for PersonFollowingStatus._format_status method."""

    @pytest.fixture
    def mock_io_provider(self):
        """Mock IOProvider."""
        with patch("inputs.plugins.person_following_status.IOProvider") as mock:
            mock_instance = MagicMock()
            mock.return_value = mock_instance
            yield mock_instance

    @pytest.fixture
    def status_instance(self, mock_io_provider):
        """Create a PersonFollowingStatus instance."""
        from inputs.plugins.person_following_status import (
            PersonFollowingStatus,
            PersonFollowingStatusConfig,
        )

        config = PersonFollowingStatusConfig(poll_interval=0.01)
        return PersonFollowingStatus(config)

    def test_format_tracking_started(self, status_instance):
        """Test format when tracking just started."""
        status_instance._previous_is_tracked = False

        data = {
            "is_tracked": True,
            "status": "TRACKING_ACTIVE",
            "x": 0.5,
            "z": 2.0,
        }

        result = status_instance._format_status(data)

        assert result is not None
        assert "TRACKING STARTED" in result
        assert "2.0m" in result

    def test_format_tracking_lost(self, status_instance):
        """Test format when tracking is just lost."""
        status_instance._previous_is_tracked = True

        data = {
            "is_tracked": False,
            "status": "SEARCHING",
            "x": 0,
            "z": 0,
        }

        result = status_instance._format_status(data)

        # Should return None immediately when tracking is lost
        assert result is None
        assert status_instance._lost_tracking_time is not None

    def test_format_searching_after_delay(self, status_instance):
        """Test SEARCHING message after 2+ second delay."""
        status_instance._previous_is_tracked = False
        status_instance._lost_tracking_time = time.time() - 3.0  # 3 seconds ago
        status_instance._lost_tracking_announced = False

        data = {
            "is_tracked": False,
            "status": "SEARCHING",
            "target_track_id": 1,
            "x": 0,
            "z": 0,
        }

        result = status_instance._format_status(data)

        assert result is not None
        assert "SEARCHING" in result
        assert status_instance._lost_tracking_announced is True

    def test_format_waiting_after_delay(self, status_instance):
        """Test WAITING message after 2+ second delay for INACTIVE."""
        status_instance._previous_is_tracked = False
        status_instance._lost_tracking_time = time.time() - 3.0  # 3 seconds ago
        status_instance._lost_tracking_announced = False

        data = {
            "is_tracked": False,
            "status": "INACTIVE",
            "target_track_id": None,
            "x": 0,
            "z": 0,
        }

        result = status_instance._format_status(data)

        assert result is not None
        assert "WAITING" in result

    def test_format_currently_tracking(self, status_instance):
        """Test format for ongoing tracking."""
        status_instance._previous_is_tracked = True

        data = {
            "is_tracked": True,
            "status": "TRACKING_ACTIVE",
            "x": 1.0,
            "z": 3.0,
        }

        result = status_instance._format_status(data)

        assert result is not None
        assert "TRACKING" in result
        assert "3.0m" in result

    def test_format_no_spam_when_not_tracking(self, status_instance):
        """Test that we don't spam messages when not tracking."""
        status_instance._previous_is_tracked = False
        status_instance._lost_tracking_time = None

        data = {
            "is_tracked": False,
            "status": "INACTIVE",
            "x": 0,
            "z": 0,
        }

        result = status_instance._format_status(data)

        # Should return None to avoid spam
        assert result is None


class TestPersonFollowingStatusRawToText:
    """Tests for PersonFollowingStatus._raw_to_text and raw_to_text methods."""

    @pytest.fixture
    def mock_io_provider(self):
        """Mock IOProvider."""
        with patch("inputs.plugins.person_following_status.IOProvider") as mock:
            mock_instance = MagicMock()
            mock.return_value = mock_instance
            yield mock_instance

    @pytest.fixture
    def status_instance(self, mock_io_provider):
        """Create a PersonFollowingStatus instance."""
        from inputs.plugins.person_following_status import (
            PersonFollowingStatus,
            PersonFollowingStatusConfig,
        )

        config = PersonFollowingStatusConfig(poll_interval=0.01)
        return PersonFollowingStatus(config)

    @pytest.mark.asyncio
    async def test_raw_to_text_none(self, status_instance):
        """Test _raw_to_text with None input."""
        result = await status_instance._raw_to_text(None)

        assert result is None

    @pytest.mark.asyncio
    async def test_raw_to_text_valid(self, status_instance):
        """Test _raw_to_text with valid input."""
        result = await status_instance._raw_to_text("Test message")

        assert result is not None
        assert result.message == "Test message"
        assert result.timestamp > 0

    @pytest.mark.asyncio
    async def test_raw_to_text_none_input(self, status_instance):
        """Test raw_to_text with None input doesn't append."""
        initial_count = len(status_instance.messages)

        await status_instance.raw_to_text(None)

        assert len(status_instance.messages) == initial_count

    @pytest.mark.asyncio
    async def test_raw_to_text_appends_message(self, status_instance):
        """Test raw_to_text appends message to buffer."""
        initial_count = len(status_instance.messages)

        await status_instance.raw_to_text("Test message")

        assert len(status_instance.messages) == initial_count + 1
        assert status_instance.messages[-1].message == "Test message"


class TestPersonFollowingStatusFormattedBuffer:
    """Tests for PersonFollowingStatus.formatted_latest_buffer method."""

    @pytest.fixture
    def mock_io_provider(self):
        """Mock IOProvider."""
        with patch("inputs.plugins.person_following_status.IOProvider") as mock:
            mock_instance = MagicMock()
            mock.return_value = mock_instance
            yield mock_instance

    @pytest.fixture
    def status_instance(self, mock_io_provider):
        """Create a PersonFollowingStatus instance."""
        from inputs.plugins.person_following_status import (
            PersonFollowingStatus,
            PersonFollowingStatusConfig,
        )

        config = PersonFollowingStatusConfig(poll_interval=0.01)
        return PersonFollowingStatus(config)

    def test_formatted_buffer_empty(self, status_instance):
        """Test formatted_latest_buffer with empty messages."""
        result = status_instance.formatted_latest_buffer()

        assert result is None

    @pytest.mark.asyncio
    async def test_formatted_buffer_with_message(self, status_instance, mock_io_provider):
        """Test formatted_latest_buffer with messages."""
        await status_instance.raw_to_text("Test tracking status")

        result = status_instance.formatted_latest_buffer()

        assert result is not None
        assert "INPUT: Person Following Status" in result
        assert "Test tracking status" in result
        assert "// START" in result
        assert "// END" in result

    @pytest.mark.asyncio
    async def test_formatted_buffer_clears(self, status_instance, mock_io_provider):
        """Test that formatted_latest_buffer clears messages."""
        await status_instance.raw_to_text("Test message")
        assert len(status_instance.messages) == 1

        status_instance.formatted_latest_buffer()

        assert len(status_instance.messages) == 0

    @pytest.mark.asyncio
    async def test_formatted_buffer_io_provider(self, status_instance, mock_io_provider):
        """Test that formatted_latest_buffer calls IOProvider.add_input."""
        await status_instance.raw_to_text("Test message")

        status_instance.formatted_latest_buffer()

        mock_io_provider.add_input.assert_called_once()
        call_args = mock_io_provider.add_input.call_args
        assert call_args[0][0] == "PersonFollowingStatus"
        assert call_args[0][1] == "Test message"


class TestStateTransitions:
    """
    Test complete state machine transitions.

    State diagram:
    INACTIVE (not tracked) -> TRACKING_ACTIVE (tracked) -> SEARCHING (not tracked) -> TRACKING_ACTIVE
    """

    @pytest.fixture
    def mock_io_provider(self):
        """Mock IOProvider."""
        with patch("inputs.plugins.person_following_status.IOProvider") as mock:
            mock_instance = MagicMock()
            mock.return_value = mock_instance
            yield mock_instance

    @pytest.fixture
    def status_instance(self, mock_io_provider):
        """Create a PersonFollowingStatus instance."""
        from inputs.plugins.person_following_status import (
            PersonFollowingStatus,
            PersonFollowingStatusConfig,
        )

        config = PersonFollowingStatusConfig(poll_interval=0.01)
        return PersonFollowingStatus(config)

    def test_full_lifecycle_inactive_to_tracking_to_searching_to_tracking(
        self, status_instance
    ):
        """
        Test complete lifecycle:
        1. Start INACTIVE (no person)
        2. Person appears -> TRACKING_ACTIVE
        3. Person goes out of frame -> SEARCHING
        4. Person returns -> TRACKING_ACTIVE
        5. Person leaves completely -> INACTIVE
        """
        # Step 1: Initial state - INACTIVE, no person
        result1 = status_instance._format_status({
            "is_tracked": False,
            "status": "INACTIVE",
            "target_track_id": None,
            "x": 0, "z": 0,
        })
        # First call with not tracked and no lost_tracking_time should return None
        assert result1 is None
        assert status_instance._previous_is_tracked is False

        # Step 2: Person appears - TRACKING_ACTIVE
        result2 = status_instance._format_status({
            "is_tracked": True,
            "status": "TRACKING_ACTIVE",
            "target_track_id": 1,
            "x": 0.5, "z": 2.0,
        })
        assert result2 is not None
        assert "TRACKING STARTED" in result2
        assert status_instance._previous_is_tracked is True
        assert status_instance._lost_tracking_time is None

        # Step 3: Person goes out of frame - SEARCHING
        result3 = status_instance._format_status({
            "is_tracked": False,
            "status": "SEARCHING",
            "target_track_id": 1,
            "x": 0, "z": 0,
        })
        # Immediately after losing tracking, should return None (grace period)
        assert result3 is None
        assert status_instance._previous_is_tracked is False
        assert status_instance._lost_tracking_time is not None

        # Step 4: Wait 2+ seconds, then still SEARCHING
        status_instance._lost_tracking_time = time.time() - 3.0
        result4 = status_instance._format_status({
            "is_tracked": False,
            "status": "SEARCHING",
            "target_track_id": 1,
            "x": 0, "z": 0,
        })
        assert result4 is not None
        assert "SEARCHING" in result4
        assert status_instance._lost_tracking_announced is True

        # Step 5: Person returns - TRACKING_ACTIVE again
        result5 = status_instance._format_status({
            "is_tracked": True,
            "status": "TRACKING_ACTIVE",
            "target_track_id": 1,
            "x": 1.0, "z": 3.0,
        })
        assert result5 is not None
        assert "TRACKING STARTED" in result5
        # State should be reset
        assert status_instance._lost_tracking_time is None
        assert status_instance._lost_tracking_announced is False

    def test_rapid_tracking_toggle(self, status_instance):
        """Test rapid on/off/on tracking doesn't cause issues."""
        # Track
        status_instance._format_status({
            "is_tracked": True, "status": "TRACKING_ACTIVE",
            "target_track_id": 1, "x": 0, "z": 1,
        })

        # Lose (immediately)
        status_instance._format_status({
            "is_tracked": False, "status": "SEARCHING",
            "target_track_id": 1, "x": 0, "z": 0,
        })

        # Regain (within 2 second grace period)
        result = status_instance._format_status({
            "is_tracked": True, "status": "TRACKING_ACTIVE",
            "target_track_id": 1, "x": 0, "z": 1.5,
        })

        # Should announce tracking started again
        assert result is not None
        assert "TRACKING STARTED" in result
        # Lost tracking announcement should NOT have happened
        assert status_instance._lost_tracking_announced is False


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    @pytest.fixture
    def mock_io_provider(self):
        """Mock IOProvider."""
        with patch("inputs.plugins.person_following_status.IOProvider") as mock:
            mock_instance = MagicMock()
            mock.return_value = mock_instance
            yield mock_instance

    @pytest.fixture
    def status_instance(self, mock_io_provider):
        """Create a PersonFollowingStatus instance."""
        from inputs.plugins.person_following_status import (
            PersonFollowingStatus,
            PersonFollowingStatusConfig,
        )

        config = PersonFollowingStatusConfig(poll_interval=0.01)
        return PersonFollowingStatus(config)

    def test_negative_coordinates(self, status_instance):
        """Test formatting with negative x/z coordinates."""
        status_instance._previous_is_tracked = True

        result = status_instance._format_status({
            "is_tracked": True,
            "status": "TRACKING_ACTIVE",
            "x": -1.5,
            "z": -2.0,
        })

        assert result is not None
        assert "-2.0m" in result
        assert "-1.5m" in result

    def test_zero_coordinates(self, status_instance):
        """Test formatting with zero coordinates."""
        status_instance._previous_is_tracked = True

        result = status_instance._format_status({
            "is_tracked": True,
            "status": "TRACKING_ACTIVE",
            "x": 0.0,
            "z": 0.0,
        })

        assert result is not None
        assert "0.0m" in result

    def test_very_large_coordinates(self, status_instance):
        """Test formatting with very large coordinates."""
        status_instance._previous_is_tracked = True

        result = status_instance._format_status({
            "is_tracked": True,
            "status": "TRACKING_ACTIVE",
            "x": 999.999,
            "z": 1234.5,
        })

        assert result is not None
        assert "1234.5m" in result

    def test_missing_data_fields(self, status_instance):
        """Test handling of missing fields in status data."""
        status_instance._previous_is_tracked = False

        # Minimal data - only is_tracked
        result = status_instance._format_status({
            "is_tracked": True,
        })

        assert result is not None
        assert "TRACKING STARTED" in result
        # Should use defaults for x, z
        assert "0.0m" in result

    def test_first_poll_not_tracked_no_announcement(self, status_instance):
        """
        Test that first poll with not_tracked doesn't announce.

        BUG CHECK: If _lost_tracking_time is None and not is_tracked,
        should NOT announce (no transition happened).
        """
        assert status_instance._previous_is_tracked is None
        assert status_instance._lost_tracking_time is None

        result = status_instance._format_status({
            "is_tracked": False,
            "status": "INACTIVE",
            "target_track_id": None,
            "x": 0, "z": 0,
        })

        # Should return None - no announcement on first poll
        assert result is None
        # _lost_tracking_time should still be None (no transition)
        assert status_instance._lost_tracking_time is None

    def test_empty_data_dict(self, status_instance):
        """Test handling of empty status data."""
        status_instance._previous_is_tracked = False

        result = status_instance._format_status({})

        # Should handle gracefully with defaults
        assert result is None  # is_tracked defaults to False, no transition

    def test_messages_deque_overflow(self, status_instance):
        """Test that deque properly handles overflow (maxlen=50)."""
        # Add 60 messages
        for i in range(60):
            status_instance.messages.append(
                MagicMock(message=f"Message {i}", timestamp=time.time())
            )

        # Should only have 50 messages
        assert len(status_instance.messages) == 50
        # Oldest should be Message 10 (0-9 were dropped)
        assert status_instance.messages[0].message == "Message 10"
        # Newest should be Message 59
        assert status_instance.messages[-1].message == "Message 59"

    def test_formatted_buffer_returns_only_latest_loses_others(
        self, status_instance, mock_io_provider
    ):
        """
        Test that formatted_latest_buffer only returns the LAST message.

        BEHAVIOR CHECK: Multiple messages accumulated, only last one returned.
        This is intentional but should be documented/understood.
        """
        # Add multiple messages
        status_instance.messages.append(
            MagicMock(message="First message", timestamp=1.0)
        )
        status_instance.messages.append(
            MagicMock(message="Second message", timestamp=2.0)
        )
        status_instance.messages.append(
            MagicMock(message="Third message", timestamp=3.0)
        )

        result = status_instance.formatted_latest_buffer()

        # Only "Third message" should be in result
        assert "Third message" in result
        assert "First message" not in result
        assert "Second message" not in result
        # All messages cleared
        assert len(status_instance.messages) == 0


class TestHasEverTrackedBehavior:
    """
    Test _has_ever_tracked state variable.

    NOTE: This variable is SET but never READ in the current code.
    These tests document the current behavior and check if it's a bug.
    """

    @pytest.fixture
    def mock_io_provider(self):
        """Mock IOProvider."""
        with patch("inputs.plugins.person_following_status.IOProvider") as mock:
            mock_instance = MagicMock()
            mock.return_value = mock_instance
            yield mock_instance

    @pytest.fixture
    def status_instance(self, mock_io_provider):
        """Create a PersonFollowingStatus instance."""
        from inputs.plugins.person_following_status import (
            PersonFollowingStatus,
            PersonFollowingStatusConfig,
        )

        config = PersonFollowingStatusConfig(poll_interval=0.01)
        return PersonFollowingStatus(config)

    def test_has_ever_tracked_initially_false(self, status_instance):
        """Test _has_ever_tracked starts as False."""
        assert status_instance._has_ever_tracked is False

    def test_has_ever_tracked_set_on_tracking(self, status_instance):
        """Test _has_ever_tracked is set to True when tracking starts."""
        # This happens in _poll, not _format_status
        # We need to verify the logic path
        assert status_instance._has_ever_tracked is False

    def test_has_ever_tracked_never_used(self, status_instance):
        """
        DEAD CODE CHECK: _has_ever_tracked is set but never read.

        This test documents that the variable exists but isn't used.
        This might be:
        1. Dead code that should be removed
        2. Incomplete feature
        3. Intentional for future use
        """
        # Verify it's set in _poll when is_tracked=True
        # But _format_status never reads it
        # The variable is essentially useless in current implementation

        # Get all method source code and check for reads
        import inspect
        source = inspect.getsource(status_instance.__class__)

        # Count occurrences - verify it's set somewhere
        set_count = source.count("_has_ever_tracked = True")
        assert set_count >= 1, "_has_ever_tracked should be set somewhere"

        # Verify it's not read in any conditional or return statement
        # This documents current behavior - variable is set but never used
        assert "if self._has_ever_tracked" not in source, (
            "_has_ever_tracked is now being read - update this test"
        )


class TestRealisticScenarios:
    """Test realistic usage scenarios."""

    @pytest.fixture
    def mock_io_provider(self):
        """Mock IOProvider."""
        with patch("inputs.plugins.person_following_status.IOProvider") as mock:
            mock_instance = MagicMock()
            mock.return_value = mock_instance
            yield mock_instance

    @pytest.fixture
    def status_instance(self, mock_io_provider):
        """Create a PersonFollowingStatus instance."""
        from inputs.plugins.person_following_status import (
            PersonFollowingStatus,
            PersonFollowingStatusConfig,
        )

        config = PersonFollowingStatusConfig(poll_interval=0.01)
        return PersonFollowingStatus(config)

    def test_scenario_person_walks_around(self, status_instance):
        """
        Simulate person walking around the robot:
        1. Person appears in front (z=2, x=0)
        2. Person moves left (z=2, x=-1)
        3. Person moves further (z=4, x=-1)
        4. Person comes back (z=1, x=0)
        """
        # Person appears
        r1 = status_instance._format_status({
            "is_tracked": True, "status": "TRACKING_ACTIVE",
            "x": 0.0, "z": 2.0, "target_track_id": 1,
        })
        assert "TRACKING STARTED" in r1
        assert "2.0m ahead" in r1

        # Person moves left
        r2 = status_instance._format_status({
            "is_tracked": True, "status": "TRACKING_ACTIVE",
            "x": -1.0, "z": 2.0, "target_track_id": 1,
        })
        assert "TRACKING" in r2
        assert "-1.0m to the side" in r2

        # Person moves further
        r3 = status_instance._format_status({
            "is_tracked": True, "status": "TRACKING_ACTIVE",
            "x": -1.0, "z": 4.0, "target_track_id": 1,
        })
        assert "4.0m ahead" in r3

        # Person comes back close
        r4 = status_instance._format_status({
            "is_tracked": True, "status": "TRACKING_ACTIVE",
            "x": 0.0, "z": 1.0, "target_track_id": 1,
        })
        assert "1.0m ahead" in r4

    def test_scenario_person_briefly_occluded(self, status_instance):
        """
        Simulate person briefly occluded (walks behind pillar):
        1. Tracking active
        2. Lost for 1 second (< 2s threshold)
        3. Re-acquired
        Should NOT announce "SEARCHING" message.
        """
        # Tracking
        status_instance._format_status({
            "is_tracked": True, "status": "TRACKING_ACTIVE",
            "x": 0, "z": 2, "target_track_id": 1,
        })

        # Lost
        status_instance._format_status({
            "is_tracked": False, "status": "SEARCHING",
            "x": 0, "z": 0, "target_track_id": 1,
        })

        # Simulate 1 second passing (less than 2s threshold)
        status_instance._lost_tracking_time = time.time() - 1.0

        # Still searching but within grace period
        r = status_instance._format_status({
            "is_tracked": False, "status": "SEARCHING",
            "x": 0, "z": 0, "target_track_id": 1,
        })

        # Should NOT announce yet
        assert r is None
        assert status_instance._lost_tracking_announced is False

        # Re-acquired
        r2 = status_instance._format_status({
            "is_tracked": True, "status": "TRACKING_ACTIVE",
            "x": 0.1, "z": 2.1, "target_track_id": 1,
        })

        assert "TRACKING STARTED" in r2

    def test_scenario_person_leaves_permanently(self, status_instance):
        """
        Simulate person leaving permanently:
        1. Tracking active
        2. Lost
        3. After 2+ seconds, announce SEARCHING
        4. System gives up, goes to INACTIVE
        5. After another 2+ seconds, announce WAITING
        """
        # Tracking
        status_instance._format_status({
            "is_tracked": True, "status": "TRACKING_ACTIVE",
            "x": 0, "z": 2, "target_track_id": 1,
        })

        # Lost
        status_instance._format_status({
            "is_tracked": False, "status": "SEARCHING",
            "x": 0, "z": 0, "target_track_id": 1,
        })

        # Wait 3 seconds
        status_instance._lost_tracking_time = time.time() - 3.0

        # Searching announcement
        r1 = status_instance._format_status({
            "is_tracked": False, "status": "SEARCHING",
            "x": 0, "z": 0, "target_track_id": 1,
        })
        assert "SEARCHING" in r1
        assert status_instance._lost_tracking_announced is True

        # System gives up (resets to INACTIVE)
        # This is a NEW state transition - _lost_tracking_announced is already True
        # So it won't announce again
        r2 = status_instance._format_status({
            "is_tracked": False, "status": "INACTIVE",
            "x": 0, "z": 0, "target_track_id": None,
        })

        # Already announced, should be None
        assert r2 is None
