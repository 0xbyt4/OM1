import logging
import queue
import threading
from typing import Callable, Optional, Union

import numpy as np
import sounddevice as sd
import torch

from .singleton import singleton


@singleton
class ChatterboxTTSProvider:
    """
    Text-to-Speech Provider using Chatterbox for local inference.

    A singleton class that handles text-to-speech conversion using the Chatterbox
    model from Resemble AI. Supports Turbo (350M), Standard (500M), and
    Multilingual (500M, 23+ languages) variants with optional zero-shot voice
    cloning via audio prompts.
    """

    def __init__(
        self,
        model_variant: str = "standard",
        device: str = "auto",
        voice_prompt_path: Optional[str] = None,
        exaggeration: float = 0.5,
        cfg_weight: float = 0.5,
        language_id: Optional[str] = None,
    ):
        """
        Initialize the ChatterboxTTSProvider instance.

        Parameters
        ----------
        model_variant : str
            Model variant to use. "standard" for ChatterboxTTS (500M, English),
            "multilingual" for ChatterboxMultilingualTTS (500M, 23+ languages).
            Defaults to "standard".
        device : str
            Device to run inference on. "auto" selects CUDA if available,
            otherwise MPS, otherwise CPU. Defaults to "auto".
        voice_prompt_path : str, optional
            Path to a reference audio file for zero-shot voice cloning.
            If None, uses the default voice. Defaults to None.
        exaggeration : float
            Controls expressiveness of the generated speech.
            Higher values produce more expressive speech. Defaults to 0.5.
        cfg_weight : float
            Classifier-free guidance weight for generation quality.
            Defaults to 0.5.
        language_id : str, optional
            Language code for multilingual model (e.g., "en", "tr", "fr", "zh").
            Only used with the "multilingual" variant. Defaults to None.
        """
        self._model_variant = model_variant
        self._device = self._resolve_device(device)
        self._voice_prompt_path = voice_prompt_path
        self._exaggeration = exaggeration
        self._cfg_weight = cfg_weight
        self._language_id = language_id

        self.running: bool = False
        self._model = None
        self._sample_rate: int = 24000
        self._request_queue: queue.Queue = queue.Queue()
        self._worker_thread: Optional[threading.Thread] = None
        self._tts_state_callback: Optional[Callable] = None

    def _resolve_device(self, device: str) -> str:
        """
        Resolve the device string to an actual PyTorch device.

        Parameters
        ----------
        device : str
            Device string. "auto" will select the best available device.

        Returns
        -------
        str
            Resolved device string (cuda, mps, or cpu).
        """
        if device != "auto":
            return device

        if torch.cuda.is_available():
            return "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    def _load_model(self):
        """Load the Chatterbox model based on the configured variant."""
        if self._model is not None:
            return

        logging.info(
            f"Loading Chatterbox {self._model_variant} model on {self._device}..."
        )

        device = torch.device(self._device)

        if self._model_variant == "multilingual":
            from chatterbox.mtl_tts import ChatterboxMultilingualTTS

            self._model = ChatterboxMultilingualTTS.from_pretrained(device=device)
        else:
            from chatterbox.tts import ChatterboxTTS

            self._model = ChatterboxTTS.from_pretrained(device=device)

        self._sample_rate = self._model.sr

        logging.info(
            f"Chatterbox {self._model_variant} model loaded successfully "
            f"(sample_rate={self._sample_rate})"
        )

    def configure(
        self,
        model_variant: str = "standard",
        device: str = "auto",
        voice_prompt_path: Optional[str] = None,
        exaggeration: float = 0.5,
        cfg_weight: float = 0.5,
        language_id: Optional[str] = None,
    ):
        """
        Configure the TTS provider with given parameters.

        Parameters
        ----------
        model_variant : str
            Model variant to use. Defaults to "turbo".
        device : str
            Device for inference. Defaults to "auto".
        voice_prompt_path : str, optional
            Path to reference audio for voice cloning. Defaults to None.
        exaggeration : float
            Expressiveness control. Defaults to 0.5.
        cfg_weight : float
            Classifier-free guidance weight. Defaults to 0.5.
        language_id : str, optional
            Language code for multilingual model. Defaults to None.
        """
        resolved_device = self._resolve_device(device)
        restart_needed = (
            model_variant != self._model_variant
            or resolved_device != self._device
            or voice_prompt_path != self._voice_prompt_path
            or exaggeration != self._exaggeration
            or cfg_weight != self._cfg_weight
            or language_id != self._language_id
        )

        if not restart_needed:
            return

        if self.running:
            self.stop()

        self._model_variant = model_variant
        self._device = resolved_device
        self._voice_prompt_path = voice_prompt_path
        self._exaggeration = exaggeration
        self._cfg_weight = cfg_weight
        self._language_id = language_id
        self._model = None

        self.start()

    def register_tts_state_callback(self, tts_state_callback: Optional[Callable]):
        """
        Register a callback for TTS state changes.

        Parameters
        ----------
        tts_state_callback : Optional[Callable]
            The callback function to receive TTS state changes.
        """
        if tts_state_callback is not None:
            self._tts_state_callback = tts_state_callback

    def create_pending_message(self, text: str) -> dict:
        """
        Create a pending message for TTS processing.

        Parameters
        ----------
        text : str
            Text to be converted to speech.

        Returns
        -------
        dict
            A dictionary containing the TTS request parameters.
        """
        logging.info(f"audio_stream: {text}")
        message: dict = {
            "text": text,
            "model_variant": self._model_variant,
            "voice_prompt_path": self._voice_prompt_path,
            "exaggeration": self._exaggeration,
            "cfg_weight": self._cfg_weight,
        }
        if self._language_id:
            message["language_id"] = self._language_id
        return message

    def add_pending_message(self, message: Union[str, dict]):
        """
        Add a pending message to the TTS provider.

        Parameters
        ----------
        message : Union[str, dict]
            The message to be added, typically containing text and TTS parameters.
        """
        if not self.running:
            logging.warning(
                "TTS provider is not running. Call start() before adding messages."
            )
            return

        if isinstance(message, str):
            message = self.create_pending_message(message)

        logging.info(f"Adding pending TTS message: {message}")
        self._request_queue.put(message)

    def get_pending_message_count(self) -> int:
        """
        Get the count of pending messages in the TTS provider.

        Returns
        -------
        int
            The number of pending messages.
        """
        return self._request_queue.qsize()

    def _worker_loop(self):
        """Background worker that processes TTS requests from the queue."""
        self._load_model()

        while self.running:
            try:
                message = self._request_queue.get(timeout=0.5)
            except queue.Empty:
                continue

            try:
                self._process_message(message)
            except Exception as e:
                logging.error(f"Error processing TTS message: {e}")

    def _process_message(self, message: dict):
        """
        Process a single TTS message by running inference and playing audio.

        Parameters
        ----------
        message : dict
            The TTS request containing text and generation parameters.
        """
        text = message.get("text", "")
        if not text:
            return

        voice_prompt = message.get("voice_prompt_path", self._voice_prompt_path)
        exaggeration = message.get("exaggeration", self._exaggeration)
        cfg_weight = message.get("cfg_weight", self._cfg_weight)
        language_id = message.get("language_id", self._language_id)

        if self._tts_state_callback:
            self._tts_state_callback("speaking")

        logging.info(f"Generating speech for: {text[:50]}...")

        generate_kwargs: dict = {"text": text, "exaggeration": exaggeration}

        if voice_prompt:
            generate_kwargs["audio_prompt_path"] = voice_prompt

        if self._model_variant == "standard":
            generate_kwargs["cfg_weight"] = cfg_weight
        elif self._model_variant == "multilingual" and language_id:
            generate_kwargs["language_id"] = language_id

        if self._model is None:
            logging.error("Model not loaded, cannot generate speech")
            return

        wav = self._model.generate(**generate_kwargs)

        if isinstance(wav, torch.Tensor):
            audio_data = wav.squeeze().cpu().numpy()
        else:
            audio_data = np.array(wav)

        if audio_data.dtype != np.float32:
            audio_data = audio_data.astype(np.float32)

        if audio_data.max() > 1.0 or audio_data.min() < -1.0:
            max_val = max(abs(audio_data.max()), abs(audio_data.min()))
            if max_val > 0:
                audio_data = audio_data / max_val

        sd.play(audio_data, samplerate=self._sample_rate)
        sd.wait()

        if self._tts_state_callback:
            self._tts_state_callback("idle")

        logging.info("Speech playback completed")

    def start(self):
        """Start the TTS provider and its worker thread."""
        if self.running:
            logging.warning("Chatterbox TTS provider is already running")
            return

        self.running = True
        self._worker_thread = threading.Thread(
            target=self._worker_loop, daemon=True, name="chatterbox-tts-worker"
        )
        self._worker_thread.start()

    def stop(self):
        """Stop the TTS provider and cleanup resources."""
        if not self.running:
            logging.warning("Chatterbox TTS provider is not running")
            return

        self.running = False

        if self._worker_thread and self._worker_thread.is_alive():
            self._worker_thread.join(timeout=5.0)

        while not self._request_queue.empty():
            try:
                self._request_queue.get_nowait()
            except queue.Empty:
                break

        self._model = None
