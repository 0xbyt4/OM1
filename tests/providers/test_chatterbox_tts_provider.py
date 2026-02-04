import sys
from unittest.mock import MagicMock, Mock, patch

import pytest
import torch

from providers.chatterbox_tts_provider import ChatterboxTTSProvider


@pytest.fixture(autouse=True, scope="session")
def mock_chatterbox_modules():
    """Mock all chatterbox modules before any imports."""
    mock_chatterbox = MagicMock()
    mock_tts = MagicMock()
    mock_tts_turbo = MagicMock()
    mock_mtl_tts = MagicMock()
    mock_chatterbox.tts = mock_tts
    mock_chatterbox.tts_turbo = mock_tts_turbo
    mock_chatterbox.mtl_tts = mock_mtl_tts
    sys.modules["chatterbox"] = mock_chatterbox
    sys.modules["chatterbox.tts"] = mock_tts
    sys.modules["chatterbox.tts_turbo"] = mock_tts_turbo
    sys.modules["chatterbox.mtl_tts"] = mock_mtl_tts
    yield {
        "tts": mock_tts,
        "tts_turbo": mock_tts_turbo,
        "mtl_tts": mock_mtl_tts,
    }
    for mod in [
        "chatterbox",
        "chatterbox.tts",
        "chatterbox.tts_turbo",
        "chatterbox.mtl_tts",
    ]:
        if mod in sys.modules:
            del sys.modules[mod]


@pytest.fixture(autouse=True)
def reset_singleton(mock_chatterbox_modules):
    """Reset the singleton instance before each test."""
    for mock_mod in mock_chatterbox_modules.values():
        mock_mod.reset_mock()
    ChatterboxTTSProvider.reset()
    yield
    ChatterboxTTSProvider.reset()


class TestChatterboxTTSProviderInit:
    """Test initialization of the provider."""

    def test_default_init(self):
        """Test default initialization values."""
        provider = ChatterboxTTSProvider()

        assert provider._model_variant == "standard"
        assert provider._voice_prompt_path is None
        assert provider._exaggeration == 0.5
        assert provider._cfg_weight == 0.5
        assert provider._language_id is None
        assert provider._sample_rate == 24000
        assert provider.running is False
        assert provider._model is None

    def test_custom_init(self):
        """Test custom initialization values."""
        provider = ChatterboxTTSProvider(
            model_variant="multilingual",
            device="cpu",
            voice_prompt_path="/path/to/voice.wav",
            exaggeration=0.8,
            cfg_weight=0.7,
            language_id="tr",
        )

        assert provider._model_variant == "multilingual"
        assert provider._device == "cpu"
        assert provider._voice_prompt_path == "/path/to/voice.wav"
        assert provider._exaggeration == 0.8
        assert provider._cfg_weight == 0.7
        assert provider._language_id == "tr"

    def test_singleton_pattern(self):
        """Test that only one instance is created."""
        provider1 = ChatterboxTTSProvider()
        provider2 = ChatterboxTTSProvider()

        assert provider1 is provider2


class TestDeviceResolution:
    """Test device resolution logic."""

    def test_explicit_device(self):
        """Test that explicit device is used as-is."""
        provider = ChatterboxTTSProvider(device="cpu")
        assert provider._device == "cpu"

    @patch("providers.chatterbox_tts_provider.torch")
    def test_auto_device_cuda(self, mock_torch):
        """Test auto device selects CUDA when available."""
        mock_torch.cuda.is_available.return_value = True
        provider = ChatterboxTTSProvider(device="auto")
        assert provider._device == "cuda"

    @patch("providers.chatterbox_tts_provider.torch")
    def test_auto_device_mps(self, mock_torch):
        """Test auto device selects MPS when CUDA not available."""
        mock_torch.cuda.is_available.return_value = False
        mock_torch.backends.mps.is_available.return_value = True
        provider = ChatterboxTTSProvider(device="auto")
        assert provider._device == "mps"

    @patch("providers.chatterbox_tts_provider.torch")
    def test_auto_device_cpu_fallback(self, mock_torch):
        """Test auto device falls back to CPU."""
        mock_torch.cuda.is_available.return_value = False
        mock_torch.backends.mps.is_available.return_value = False
        provider = ChatterboxTTSProvider(device="auto")
        assert provider._device == "cpu"


class TestModelLoading:
    """Test model loading behavior."""

    @patch("providers.chatterbox_tts_provider.torch")
    def test_load_standard_model(self, mock_torch, mock_chatterbox_modules):
        """Test loading the standard model variant."""
        mock_torch.cuda.is_available.return_value = False
        mock_torch.backends.mps.is_available.return_value = False
        mock_model = Mock()
        mock_model.sr = 24000
        mock_chatterbox_modules["tts"].ChatterboxTTS.from_pretrained.return_value = (
            mock_model
        )
        provider = ChatterboxTTSProvider(model_variant="standard", device="cpu")

        provider._load_model()
        mock_chatterbox_modules[
            "tts"
        ].ChatterboxTTS.from_pretrained.assert_called_once_with(device="cpu")
        assert provider._sample_rate == 24000

    @patch("providers.chatterbox_tts_provider.torch")
    def test_load_multilingual_model(self, mock_torch, mock_chatterbox_modules):
        """Test loading the multilingual model variant."""
        mock_torch.cuda.is_available.return_value = False
        mock_torch.backends.mps.is_available.return_value = False
        mock_model = Mock()
        mock_model.sr = 24000
        mock_chatterbox_modules[
            "mtl_tts"
        ].ChatterboxMultilingualTTS.from_pretrained.return_value = mock_model
        provider = ChatterboxTTSProvider(model_variant="multilingual", device="cpu")

        provider._load_model()
        mock_chatterbox_modules[
            "mtl_tts"
        ].ChatterboxMultilingualTTS.from_pretrained.assert_called_once_with(
            device="cpu"
        )

    @patch("providers.chatterbox_tts_provider.torch")
    def test_load_model_skips_if_already_loaded(
        self, mock_torch, mock_chatterbox_modules
    ):
        """Test that model is not reloaded if already present."""
        mock_torch.cuda.is_available.return_value = False
        mock_torch.backends.mps.is_available.return_value = False
        provider = ChatterboxTTSProvider(device="cpu")
        provider._model = Mock()

        provider._load_model()
        mock_chatterbox_modules["tts"].ChatterboxTTS.from_pretrained.assert_not_called()


class TestPendingMessages:
    """Test message creation and queueing."""

    def test_create_pending_message(self):
        """Test creating a pending message."""
        provider = ChatterboxTTSProvider()
        message = provider.create_pending_message("Hello world")

        assert message["text"] == "Hello world"
        assert message["model_variant"] == "standard"
        assert message["voice_prompt_path"] is None
        assert message["exaggeration"] == 0.5
        assert message["cfg_weight"] == 0.5
        assert "language_id" not in message

    def test_create_pending_message_with_custom_settings(self):
        """Test creating a pending message with custom provider settings."""
        provider = ChatterboxTTSProvider(
            model_variant="multilingual",
            voice_prompt_path="/path/to/voice.wav",
            exaggeration=0.8,
            cfg_weight=0.7,
            language_id="tr",
        )
        message = provider.create_pending_message("Test text")

        assert message["text"] == "Test text"
        assert message["model_variant"] == "multilingual"
        assert message["voice_prompt_path"] == "/path/to/voice.wav"
        assert message["exaggeration"] == 0.8
        assert message["cfg_weight"] == 0.7
        assert message["language_id"] == "tr"

    def test_add_pending_message_when_running(self):
        """Test adding a message when provider is running."""
        provider = ChatterboxTTSProvider()
        provider.running = True
        provider.add_pending_message({"text": "Hello"})

        assert provider._request_queue.qsize() == 1

    def test_add_pending_message_when_not_running(self):
        """Test adding a message when provider is not running."""
        provider = ChatterboxTTSProvider()
        provider.add_pending_message({"text": "Hello"})

        assert provider._request_queue.qsize() == 0

    def test_add_pending_message_string(self):
        """Test adding a string message auto-converts to dict."""
        provider = ChatterboxTTSProvider()
        provider.running = True
        provider.add_pending_message("Hello world")

        assert provider._request_queue.qsize() == 1
        message = provider._request_queue.get_nowait()
        assert message["text"] == "Hello world"

    def test_get_pending_message_count(self):
        """Test getting the count of pending messages."""
        provider = ChatterboxTTSProvider()
        provider.running = True

        assert provider.get_pending_message_count() == 0

        provider.add_pending_message({"text": "Hello"})
        assert provider.get_pending_message_count() == 1

        provider.add_pending_message({"text": "World"})
        assert provider.get_pending_message_count() == 2


class TestProcessMessage:
    """Test message processing and audio playback."""

    @patch("providers.chatterbox_tts_provider.sd")
    def test_process_message_tensor_output(self, mock_sd):
        """Test processing a message with tensor output."""
        provider = ChatterboxTTSProvider()
        mock_model = Mock()
        audio_tensor = torch.randn(1, 24000)
        mock_model.generate.return_value = audio_tensor
        provider._model = mock_model

        provider._process_message({"text": "Hello", "exaggeration": 0.5})

        mock_model.generate.assert_called_once()
        mock_sd.play.assert_called_once()
        mock_sd.wait.assert_called_once()

    @patch("providers.chatterbox_tts_provider.sd")
    def test_process_message_empty_text(self, mock_sd):
        """Test processing a message with empty text."""
        provider = ChatterboxTTSProvider()
        provider._model = Mock()

        provider._process_message({"text": ""})

        provider._model.generate.assert_not_called()
        mock_sd.play.assert_not_called()

    @patch("providers.chatterbox_tts_provider.sd")
    def test_process_message_with_voice_prompt(self, mock_sd):
        """Test processing a message with voice prompt for cloning."""
        provider = ChatterboxTTSProvider()
        mock_model = Mock()
        audio_tensor = torch.randn(1, 24000)
        mock_model.generate.return_value = audio_tensor
        provider._model = mock_model

        provider._process_message(
            {
                "text": "Hello",
                "voice_prompt_path": "/path/to/voice.wav",
                "exaggeration": 0.5,
            }
        )

        call_kwargs = mock_model.generate.call_args[1]
        assert call_kwargs["audio_prompt_path"] == "/path/to/voice.wav"

    @patch("providers.chatterbox_tts_provider.sd")
    def test_process_message_callback(self, mock_sd):
        """Test that TTS state callback is called during processing."""
        provider = ChatterboxTTSProvider()
        mock_model = Mock()
        audio_tensor = torch.randn(1, 24000)
        mock_model.generate.return_value = audio_tensor
        provider._model = mock_model

        callback = Mock()
        provider._tts_state_callback = callback

        provider._process_message({"text": "Hello", "exaggeration": 0.5})

        callback.assert_any_call("speaking")
        callback.assert_any_call("idle")

    @patch("providers.chatterbox_tts_provider.sd")
    def test_process_message_normalizes_audio(self, mock_sd):
        """Test that audio exceeding [-1, 1] range is normalized."""
        provider = ChatterboxTTSProvider()
        mock_model = Mock()
        audio_tensor = torch.tensor([[2.0, -3.0, 1.5]])
        mock_model.generate.return_value = audio_tensor
        provider._model = mock_model

        provider._process_message({"text": "Hello", "exaggeration": 0.5})

        played_audio = mock_sd.play.call_args[0][0]
        assert played_audio.max() <= 1.0
        assert played_audio.min() >= -1.0


class TestStartStop:
    """Test start and stop behavior."""

    def test_start(self):
        """Test starting the provider."""
        provider = ChatterboxTTSProvider()

        with patch.object(provider, "_worker_loop"):
            provider.start()

            assert provider.running is True
            assert provider._worker_thread is not None

            provider.running = False
            if provider._worker_thread.is_alive():
                provider._worker_thread.join(timeout=1.0)

    def test_start_already_running(self):
        """Test starting when already running does nothing."""
        provider = ChatterboxTTSProvider()
        provider.running = True

        provider.start()

        assert provider._worker_thread is None

    def test_stop(self):
        """Test stopping the provider."""
        provider = ChatterboxTTSProvider()
        provider.running = True
        provider._model = Mock()
        provider._worker_thread = Mock()
        provider._worker_thread.is_alive.return_value = False

        provider.stop()

        assert provider.running is False
        assert provider._model is None

    def test_stop_not_running(self):
        """Test stopping when not running does nothing."""
        provider = ChatterboxTTSProvider()
        provider.running = False

        provider.stop()

        assert provider.running is False

    def test_stop_clears_queue(self):
        """Test that stop clears the request queue."""
        provider = ChatterboxTTSProvider()
        provider.running = True
        provider._request_queue.put({"text": "Hello"})
        provider._request_queue.put({"text": "World"})
        provider._worker_thread = Mock()
        provider._worker_thread.is_alive.return_value = False

        provider.stop()

        assert provider._request_queue.empty()


class TestConfigure:
    """Test reconfiguration behavior."""

    def test_configure_no_change(self):
        """Test configure with no changes does nothing."""
        provider = ChatterboxTTSProvider(device="cpu")

        with patch.object(provider, "stop") as mock_stop:
            provider.configure(device="cpu")
            mock_stop.assert_not_called()

    def test_configure_with_changes(self):
        """Test configure with changes restarts the provider."""
        provider = ChatterboxTTSProvider(device="cpu")
        provider.running = True
        provider._worker_thread = Mock()
        provider._worker_thread.is_alive.return_value = False

        with patch.object(provider, "start"):
            provider.configure(
                model_variant="standard",
                device="cpu",
                exaggeration=0.9,
            )

            assert provider._model_variant == "standard"
            assert provider._exaggeration == 0.9
            assert provider._model is None

    def test_register_tts_state_callback(self):
        """Test registering a TTS state callback."""
        provider = ChatterboxTTSProvider()
        callback = Mock()

        provider.register_tts_state_callback(callback)

        assert provider._tts_state_callback == callback

    def test_register_tts_state_callback_none(self):
        """Test registering None callback does nothing."""
        provider = ChatterboxTTSProvider()
        provider._tts_state_callback = Mock()

        provider.register_tts_state_callback(None)

        assert provider._tts_state_callback is not None


if __name__ == "__main__":
    pytest.main([__file__])
