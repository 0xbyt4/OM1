import sys
import time
from unittest.mock import MagicMock, Mock, patch

import pytest

from actions.speak.connector.chatterbox_tts import (  # noqa: E402
    SpeakChatterboxTTSConfig,
    SpeakChatterboxTTSConnector,
)
from actions.speak.interface import SpeakInput  # noqa: E402
from zenoh_msgs import AudioStatus, String  # noqa: E402


@pytest.fixture(autouse=True, scope="session")
def mock_zenoh_module():
    """Mock the zenoh module before any imports."""
    mock_zenoh = MagicMock()
    mock_zenoh.Sample = MagicMock
    sys.modules["zenoh"] = mock_zenoh
    yield mock_zenoh
    if "zenoh" in sys.modules:
        del sys.modules["zenoh"]


@pytest.fixture
def default_config():
    """Create a default config for testing."""
    return SpeakChatterboxTTSConfig()


@pytest.fixture
def custom_config():
    """Create a custom config for testing."""
    return SpeakChatterboxTTSConfig(
        model_variant="multilingual",
        device="cpu",
        voice_prompt_path="/path/to/voice.wav",
        exaggeration=0.8,
        cfg_weight=0.7,
        language_id="tr",
        enable_tts_interrupt=True,
        silence_rate=2,
        api_key="test_api_key",  # type: ignore
    )


@pytest.fixture
def speak_input():
    """Create a SpeakInput instance for testing."""
    return SpeakInput(action="Hello, world!")


@pytest.fixture
def mock_zenoh_session():
    """Create a mock Zenoh session."""
    session = Mock()
    session.declare_publisher.return_value = Mock()
    session.declare_subscriber.return_value = Mock()
    session.close = Mock()
    return session


class TestSpeakChatterboxTTSConfig:
    """Test the configuration class."""

    def test_default_config(self):
        """Test default configuration values."""
        config = SpeakChatterboxTTSConfig()

        assert config.model_variant == "standard"
        assert config.device == "auto"
        assert config.voice_prompt_path is None
        assert config.exaggeration == 0.5
        assert config.cfg_weight == 0.5
        assert config.language_id is None
        assert config.enable_tts_interrupt is False
        assert config.silence_rate == 0

    def test_custom_config(self):
        """Test custom configuration values."""
        config = SpeakChatterboxTTSConfig(
            model_variant="multilingual",
            device="cuda",
            voice_prompt_path="/path/to/voice.wav",
            exaggeration=0.8,
            cfg_weight=0.7,
            language_id="fr",
            enable_tts_interrupt=True,
            silence_rate=5,
        )

        assert config.model_variant == "multilingual"
        assert config.device == "cuda"
        assert config.voice_prompt_path == "/path/to/voice.wav"
        assert config.exaggeration == 0.8
        assert config.cfg_weight == 0.7
        assert config.language_id == "fr"
        assert config.enable_tts_interrupt is True
        assert config.silence_rate == 5


class TestSpeakChatterboxTTSConnector:
    """Test the Chatterbox TTS connector."""

    @patch("actions.speak.connector.chatterbox_tts.open_zenoh_session")
    @patch("actions.speak.connector.chatterbox_tts.ChatterboxTTSProvider")
    @patch("actions.speak.connector.chatterbox_tts.IOProvider")
    @patch("actions.speak.connector.chatterbox_tts.TeleopsConversationProvider")
    def test_init_with_default_config(
        self,
        mock_conversation_provider,
        mock_io_provider,
        mock_tts_provider,
        mock_open_zenoh_session,
        default_config,
        mock_zenoh_session,
    ):
        """Test initialization with default configuration."""
        mock_open_zenoh_session.return_value = mock_zenoh_session
        mock_tts_instance = Mock()
        mock_tts_provider.return_value = mock_tts_instance

        connector = SpeakChatterboxTTSConnector(default_config)

        mock_open_zenoh_session.assert_called_once()
        assert mock_zenoh_session.declare_publisher.call_count == 2
        assert mock_zenoh_session.declare_subscriber.call_count == 2

        mock_tts_provider.assert_called_once_with(
            model_variant="standard",
            device="auto",
            voice_prompt_path=None,
            exaggeration=0.5,
            cfg_weight=0.5,
            language_id=None,
        )

        mock_tts_instance.start.assert_called_once()
        mock_tts_instance.configure.assert_called_once()

        assert connector.silence_rate == 0
        assert connector.silence_counter == 0
        assert connector.tts_enabled is True
        assert connector.session == mock_zenoh_session

    @patch("actions.speak.connector.chatterbox_tts.open_zenoh_session")
    @patch("actions.speak.connector.chatterbox_tts.ChatterboxTTSProvider")
    @patch("actions.speak.connector.chatterbox_tts.IOProvider")
    @patch("actions.speak.connector.chatterbox_tts.TeleopsConversationProvider")
    def test_init_with_custom_config(
        self,
        mock_conversation_provider,
        mock_io_provider,
        mock_tts_provider,
        mock_open_zenoh_session,
        custom_config,
        mock_zenoh_session,
    ):
        """Test initialization with custom configuration."""
        mock_open_zenoh_session.return_value = mock_zenoh_session
        mock_tts_instance = Mock()
        mock_tts_provider.return_value = mock_tts_instance

        connector = SpeakChatterboxTTSConnector(custom_config)

        mock_tts_provider.assert_called_once_with(
            model_variant="multilingual",
            device="cpu",
            voice_prompt_path="/path/to/voice.wav",
            exaggeration=0.8,
            cfg_weight=0.7,
            language_id="tr",
        )

        assert connector.silence_rate == 2

    @patch("actions.speak.connector.chatterbox_tts.open_zenoh_session")
    @patch("actions.speak.connector.chatterbox_tts.ChatterboxTTSProvider")
    @patch("actions.speak.connector.chatterbox_tts.IOProvider")
    @patch("actions.speak.connector.chatterbox_tts.TeleopsConversationProvider")
    def test_init_zenoh_failure(
        self,
        mock_conversation_provider,
        mock_io_provider,
        mock_tts_provider,
        mock_open_zenoh_session,
        default_config,
    ):
        """Test initialization when Zenoh session fails to open."""
        mock_open_zenoh_session.side_effect = Exception("Zenoh connection failed")

        connector = SpeakChatterboxTTSConnector(default_config)

        assert connector.session is None
        assert connector.audio_pub is None

    @patch("actions.speak.connector.chatterbox_tts.open_zenoh_session")
    @patch("actions.speak.connector.chatterbox_tts.ChatterboxTTSProvider")
    @patch("actions.speak.connector.chatterbox_tts.IOProvider")
    @patch("actions.speak.connector.chatterbox_tts.TeleopsConversationProvider")
    @pytest.mark.asyncio
    async def test_connect_tts_enabled(
        self,
        mock_conversation_provider,
        mock_io_provider,
        mock_tts_provider,
        mock_open_zenoh_session,
        default_config,
        mock_zenoh_session,
        speak_input,
    ):
        """Test connect method when TTS is enabled."""
        mock_open_zenoh_session.return_value = mock_zenoh_session
        mock_tts_instance = Mock()
        mock_tts_instance.create_pending_message.return_value = {
            "id": "test_id",
            "text": "Hello, world!",
        }
        mock_tts_provider.return_value = mock_tts_instance
        mock_audio_pub = Mock()
        mock_zenoh_session.declare_publisher.return_value = mock_audio_pub

        mock_io_instance = Mock()
        mock_io_instance.llm_prompt = "INPUT: Voice: Hello"
        mock_io_provider.return_value = mock_io_instance

        mock_conversation_instance = Mock()
        mock_conversation_provider.return_value = mock_conversation_instance

        connector = SpeakChatterboxTTSConnector(default_config)
        connector.io_provider = mock_io_instance
        connector.conversation_provider = mock_conversation_instance

        await connector.connect(speak_input)

        mock_tts_instance.create_pending_message.assert_called_once_with(
            "Hello, world!"
        )
        mock_conversation_instance.store_robot_message.assert_called_once_with(
            "Hello, world!"
        )
        mock_audio_pub.put.assert_called()

    @patch("actions.speak.connector.chatterbox_tts.open_zenoh_session")
    @patch("actions.speak.connector.chatterbox_tts.ChatterboxTTSProvider")
    @patch("actions.speak.connector.chatterbox_tts.IOProvider")
    @patch("actions.speak.connector.chatterbox_tts.TeleopsConversationProvider")
    @pytest.mark.asyncio
    async def test_connect_tts_disabled(
        self,
        mock_conversation_provider,
        mock_io_provider,
        mock_tts_provider,
        mock_open_zenoh_session,
        default_config,
        mock_zenoh_session,
        speak_input,
    ):
        """Test connect method when TTS is disabled."""
        mock_open_zenoh_session.return_value = mock_zenoh_session
        mock_tts_instance = Mock()
        mock_tts_provider.return_value = mock_tts_instance

        connector = SpeakChatterboxTTSConnector(default_config)
        connector.tts_enabled = False

        await connector.connect(speak_input)

        mock_tts_instance.create_pending_message.assert_not_called()

    @patch("actions.speak.connector.chatterbox_tts.open_zenoh_session")
    @patch("actions.speak.connector.chatterbox_tts.ChatterboxTTSProvider")
    @patch("actions.speak.connector.chatterbox_tts.IOProvider")
    @patch("actions.speak.connector.chatterbox_tts.TeleopsConversationProvider")
    @pytest.mark.asyncio
    async def test_connect_silence_rate_skip(
        self,
        mock_conversation_provider,
        mock_io_provider,
        mock_tts_provider,
        mock_open_zenoh_session,
        mock_zenoh_session,
        speak_input,
    ):
        """Test connect method with silence rate causing skip."""
        config = SpeakChatterboxTTSConfig(silence_rate=2)
        mock_open_zenoh_session.return_value = mock_zenoh_session
        mock_tts_instance = Mock()
        mock_tts_instance.create_pending_message.return_value = {
            "id": "test_id",
            "text": "Hello, world!",
        }
        mock_tts_provider.return_value = mock_tts_instance

        mock_io_instance = Mock()
        mock_io_instance.llm_prompt = "INPUT: Text: Hello"
        mock_io_provider.return_value = mock_io_instance

        connector = SpeakChatterboxTTSConnector(config)
        connector.io_provider = mock_io_instance

        await connector.connect(speak_input)
        assert connector.silence_counter == 1
        mock_tts_instance.create_pending_message.assert_not_called()

        await connector.connect(speak_input)
        assert connector.silence_counter == 2
        mock_tts_instance.create_pending_message.assert_not_called()

        await connector.connect(speak_input)
        assert connector.silence_counter == 0
        mock_tts_instance.create_pending_message.assert_called_once()

    @patch("actions.speak.connector.chatterbox_tts.open_zenoh_session")
    @patch("actions.speak.connector.chatterbox_tts.ChatterboxTTSProvider")
    @patch("actions.speak.connector.chatterbox_tts.IOProvider")
    @patch("actions.speak.connector.chatterbox_tts.TeleopsConversationProvider")
    @pytest.mark.asyncio
    async def test_connect_without_audio_publisher(
        self,
        mock_conversation_provider,
        mock_io_provider,
        mock_tts_provider,
        mock_open_zenoh_session,
        default_config,
        mock_zenoh_session,
        speak_input,
    ):
        """Test connect method when audio publisher is None."""
        mock_open_zenoh_session.return_value = mock_zenoh_session
        mock_tts_instance = Mock()
        mock_tts_instance.create_pending_message.return_value = {
            "id": "test_id",
            "text": "Hello, world!",
        }
        mock_tts_provider.return_value = mock_tts_instance

        connector = SpeakChatterboxTTSConnector(default_config)
        connector.audio_pub = None

        await connector.connect(speak_input)

        mock_tts_instance.add_pending_message.assert_called_once_with(
            {"id": "test_id", "text": "Hello, world!"}
        )

    @patch("actions.speak.connector.chatterbox_tts.open_zenoh_session")
    @patch("actions.speak.connector.chatterbox_tts.ChatterboxTTSProvider")
    @patch("actions.speak.connector.chatterbox_tts.IOProvider")
    @patch("actions.speak.connector.chatterbox_tts.TeleopsConversationProvider")
    def test_zenoh_audio_message(
        self,
        mock_conversation_provider,
        mock_io_provider,
        mock_tts_provider,
        mock_open_zenoh_session,
        default_config,
        mock_zenoh_session,
    ):
        """Test processing of Zenoh audio status messages."""
        mock_open_zenoh_session.return_value = mock_zenoh_session
        mock_tts_provider.return_value = Mock()

        connector = SpeakChatterboxTTSConnector(default_config)

        mock_sample = Mock()
        mock_audio_status = Mock()
        mock_sample.payload.to_bytes.return_value = b"test_data"

        with patch(
            "actions.speak.connector.chatterbox_tts.AudioStatus"
        ) as mock_audio_status_class:
            mock_audio_status_class.deserialize.return_value = mock_audio_status

            connector.zenoh_audio_message(mock_sample)

            mock_audio_status_class.deserialize.assert_called_once_with(b"test_data")
            assert connector.audio_status == mock_audio_status

    @patch("actions.speak.connector.chatterbox_tts.open_zenoh_session")
    @patch("actions.speak.connector.chatterbox_tts.ChatterboxTTSProvider")
    @patch("actions.speak.connector.chatterbox_tts.IOProvider")
    @patch("actions.speak.connector.chatterbox_tts.TeleopsConversationProvider")
    def test_zenoh_tts_status_request_enable(
        self,
        mock_conversation_provider,
        mock_io_provider,
        mock_tts_provider,
        mock_open_zenoh_session,
        default_config,
        mock_zenoh_session,
    ):
        """Test TTS status request to enable TTS."""
        mock_open_zenoh_session.return_value = mock_zenoh_session
        mock_tts_provider.return_value = Mock()
        mock_response_pub = Mock()

        connector = SpeakChatterboxTTSConnector(default_config)
        connector._zenoh_tts_status_response_pub = mock_response_pub
        connector.tts_enabled = False

        mock_sample = Mock()
        mock_sample.payload.to_bytes.return_value = b"test_data"

        mock_header = Mock()
        mock_header.frame_id = "test_frame"

        mock_tts_status = Mock()
        mock_tts_status.code = 1  # Enable TTS
        mock_tts_status.request_id = String("test_request_id")
        mock_tts_status.header = mock_header

        with patch(
            "actions.speak.connector.chatterbox_tts.TTSStatusRequest"
        ) as mock_request_class:
            with patch("actions.speak.connector.chatterbox_tts.TTSStatusResponse"):
                mock_request_class.deserialize.return_value = mock_tts_status

                connector._zenoh_tts_status_request(mock_sample)

                assert connector.tts_enabled is True
                mock_response_pub.put.assert_called_once()

    @patch("actions.speak.connector.chatterbox_tts.open_zenoh_session")
    @patch("actions.speak.connector.chatterbox_tts.ChatterboxTTSProvider")
    @patch("actions.speak.connector.chatterbox_tts.IOProvider")
    @patch("actions.speak.connector.chatterbox_tts.TeleopsConversationProvider")
    def test_zenoh_tts_status_request_disable(
        self,
        mock_conversation_provider,
        mock_io_provider,
        mock_tts_provider,
        mock_open_zenoh_session,
        default_config,
        mock_zenoh_session,
    ):
        """Test TTS status request to disable TTS."""
        mock_open_zenoh_session.return_value = mock_zenoh_session
        mock_tts_provider.return_value = Mock()
        mock_response_pub = Mock()

        connector = SpeakChatterboxTTSConnector(default_config)
        connector._zenoh_tts_status_response_pub = mock_response_pub
        connector.tts_enabled = True

        mock_sample = Mock()
        mock_sample.payload.to_bytes.return_value = b"test_data"

        mock_header = Mock()
        mock_header.frame_id = "test_frame"

        mock_tts_status = Mock()
        mock_tts_status.code = 0  # Disable TTS
        mock_tts_status.request_id = String("test_request_id")
        mock_tts_status.header = mock_header

        with patch(
            "actions.speak.connector.chatterbox_tts.TTSStatusRequest"
        ) as mock_request_class:
            with patch("actions.speak.connector.chatterbox_tts.TTSStatusResponse"):
                mock_request_class.deserialize.return_value = mock_tts_status

                connector._zenoh_tts_status_request(mock_sample)

                assert connector.tts_enabled is False
                mock_response_pub.put.assert_called_once()

    @patch("actions.speak.connector.chatterbox_tts.open_zenoh_session")
    @patch("actions.speak.connector.chatterbox_tts.ChatterboxTTSProvider")
    @patch("actions.speak.connector.chatterbox_tts.IOProvider")
    @patch("actions.speak.connector.chatterbox_tts.TeleopsConversationProvider")
    def test_zenoh_tts_status_request_read(
        self,
        mock_conversation_provider,
        mock_io_provider,
        mock_tts_provider,
        mock_open_zenoh_session,
        default_config,
        mock_zenoh_session,
    ):
        """Test TTS status request to read current status."""
        mock_open_zenoh_session.return_value = mock_zenoh_session
        mock_tts_provider.return_value = Mock()
        mock_response_pub = Mock()

        connector = SpeakChatterboxTTSConnector(default_config)
        connector._zenoh_tts_status_response_pub = mock_response_pub
        connector.tts_enabled = True

        mock_sample = Mock()
        mock_sample.payload.to_bytes.return_value = b"test_data"

        mock_header = Mock()
        mock_header.frame_id = "test_frame"

        mock_tts_status = Mock()
        mock_tts_status.code = 2  # Read status
        mock_tts_status.request_id = String("test_request_id")
        mock_tts_status.header = mock_header

        with patch(
            "actions.speak.connector.chatterbox_tts.TTSStatusRequest"
        ) as mock_request_class:
            with patch("actions.speak.connector.chatterbox_tts.TTSStatusResponse"):
                mock_request_class.deserialize.return_value = mock_tts_status

                connector._zenoh_tts_status_request(mock_sample)

                assert connector.tts_enabled is True
                mock_response_pub.put.assert_called_once()

    @patch("actions.speak.connector.chatterbox_tts.open_zenoh_session")
    @patch("actions.speak.connector.chatterbox_tts.ChatterboxTTSProvider")
    @patch("actions.speak.connector.chatterbox_tts.IOProvider")
    @patch("actions.speak.connector.chatterbox_tts.TeleopsConversationProvider")
    def test_stop(
        self,
        mock_conversation_provider,
        mock_io_provider,
        mock_tts_provider,
        mock_open_zenoh_session,
        default_config,
        mock_zenoh_session,
    ):
        """Test stopping the connector."""
        mock_open_zenoh_session.return_value = mock_zenoh_session
        mock_tts_instance = Mock()
        mock_tts_provider.return_value = mock_tts_instance

        connector = SpeakChatterboxTTSConnector(default_config)

        connector.stop()

        mock_zenoh_session.close.assert_called_once()
        mock_tts_instance.stop.assert_called_once()

    @patch("actions.speak.connector.chatterbox_tts.open_zenoh_session")
    @patch("actions.speak.connector.chatterbox_tts.ChatterboxTTSProvider")
    @patch("actions.speak.connector.chatterbox_tts.IOProvider")
    @patch("actions.speak.connector.chatterbox_tts.TeleopsConversationProvider")
    def test_stop_no_session(
        self,
        mock_conversation_provider,
        mock_io_provider,
        mock_tts_provider,
        mock_open_zenoh_session,
        default_config,
    ):
        """Test stopping the connector when session is None."""
        mock_open_zenoh_session.side_effect = Exception("Failed to open session")
        mock_tts_instance = Mock()
        mock_tts_provider.return_value = mock_tts_instance

        connector = SpeakChatterboxTTSConnector(default_config)
        connector.stop()

        mock_tts_instance.stop.assert_called_once()

    @patch("actions.speak.connector.chatterbox_tts.open_zenoh_session")
    @patch("actions.speak.connector.chatterbox_tts.ChatterboxTTSProvider")
    @patch("actions.speak.connector.chatterbox_tts.IOProvider")
    @patch("actions.speak.connector.chatterbox_tts.TeleopsConversationProvider")
    def test_stop_no_tts(
        self,
        mock_conversation_provider,
        mock_io_provider,
        mock_tts_provider,
        mock_open_zenoh_session,
        default_config,
        mock_zenoh_session,
    ):
        """Test stopping the connector when TTS is None."""
        mock_open_zenoh_session.return_value = mock_zenoh_session
        mock_tts_instance = Mock()
        mock_tts_provider.return_value = mock_tts_instance

        connector = SpeakChatterboxTTSConnector(default_config)
        connector.tts = None  # type: ignore

        connector.stop()

        mock_zenoh_session.close.assert_called_once()

    @patch("actions.speak.connector.chatterbox_tts.open_zenoh_session")
    @patch("actions.speak.connector.chatterbox_tts.ChatterboxTTSProvider")
    @patch("actions.speak.connector.chatterbox_tts.IOProvider")
    @patch("actions.speak.connector.chatterbox_tts.TeleopsConversationProvider")
    def test_last_voice_command_time_initialization(
        self,
        mock_conversation_provider,
        mock_io_provider,
        mock_tts_provider,
        mock_open_zenoh_session,
        default_config,
        mock_zenoh_session,
    ):
        """Test that last_voice_command_time is initialized."""
        mock_open_zenoh_session.return_value = mock_zenoh_session
        mock_tts_provider.return_value = Mock()

        start_time = time.time()
        connector = SpeakChatterboxTTSConnector(default_config)
        end_time = time.time()

        assert start_time <= connector.last_voice_command_time <= end_time

    @patch("actions.speak.connector.chatterbox_tts.open_zenoh_session")
    @patch("actions.speak.connector.chatterbox_tts.ChatterboxTTSProvider")
    @patch("actions.speak.connector.chatterbox_tts.IOProvider")
    @patch("actions.speak.connector.chatterbox_tts.TeleopsConversationProvider")
    @patch("actions.speak.connector.chatterbox_tts.uuid4")
    def test_audio_status_initialization(
        self,
        mock_uuid4,
        mock_conversation_provider,
        mock_io_provider,
        mock_tts_provider,
        mock_open_zenoh_session,
        default_config,
        mock_zenoh_session,
    ):
        """Test that audio status is properly initialized."""
        mock_uuid4.return_value = "test-uuid"
        mock_open_zenoh_session.return_value = mock_zenoh_session
        mock_tts_provider.return_value = Mock()

        with patch(
            "actions.speak.connector.chatterbox_tts.prepare_header"
        ) as mock_prepare_header:
            mock_prepare_header.return_value = "test-header"

            connector = SpeakChatterboxTTSConnector(default_config)

            assert connector.audio_status is not None
            assert (
                connector.audio_status.status_speaker
                == AudioStatus.STATUS_SPEAKER.READY.value
            )
            mock_prepare_header.assert_called_with("test-uuid")


if __name__ == "__main__":
    pytest.main([__file__])
