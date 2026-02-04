import json
import logging
import time
from typing import Optional
from uuid import uuid4

import zenoh
from pydantic import Field

from actions.base import ActionConfig, ActionConnector
from actions.speak.interface import SpeakInput
from providers.chatterbox_tts_provider import ChatterboxTTSProvider
from providers.io_provider import IOProvider
from providers.teleops_conversation_provider import TeleopsConversationProvider
from zenoh_msgs import (
    AudioStatus,
    String,
    TTSStatusRequest,
    TTSStatusResponse,
    open_zenoh_session,
    prepare_header,
)


class SpeakChatterboxTTSConfig(ActionConfig):
    """
    Configuration for Chatterbox TTS connector.

    Parameters
    ----------
    model_variant : str
        Chatterbox model variant (turbo, standard, or multilingual).
    device : str
        Device for inference (auto, cuda, mps, cpu).
    voice_prompt_path : Optional[str]
        Path to reference audio for voice cloning.
    exaggeration : float
        Expressiveness control for speech generation.
    cfg_weight : float
        Classifier-free guidance weight.
    language_id : Optional[str]
        Language code for multilingual model.
    silence_rate : int
        Number of responses to skip before speaking.
    enable_tts_interrupt : bool
        Enable TTS interrupt when ASR detects speech during playback.
    """

    model_variant: str = Field(
        default="standard",
        description="Chatterbox model variant (standard or multilingual)",
    )
    device: str = Field(
        default="auto",
        description="Device for inference (auto, cuda, mps, cpu)",
    )
    voice_prompt_path: Optional[str] = Field(
        default=None,
        description="Path to reference audio for voice cloning",
    )
    exaggeration: float = Field(
        default=0.5,
        description="Expressiveness control for speech generation",
    )
    cfg_weight: float = Field(
        default=0.5,
        description="Classifier-free guidance weight",
    )
    language_id: Optional[str] = Field(
        default=None,
        description="Language code for multilingual model (e.g., en, tr, fr, zh)",
    )
    silence_rate: int = Field(
        default=0,
        description="Number of responses to skip before speaking",
    )
    enable_tts_interrupt: bool = Field(
        default=False,
        description="Enable TTS interrupt when ASR detects speech during playback",
    )


class SpeakChatterboxTTSConnector(
    ActionConnector[SpeakChatterboxTTSConfig, SpeakInput]
):
    """
    A "Speak" connector that uses the Chatterbox TTS Provider to perform Text-to-Speech.
    This connector is compatible with the standard SpeakInput interface.
    """

    def __init__(self, config: SpeakChatterboxTTSConfig):
        """
        Initializes the connector and its underlying TTS provider.

        Parameters
        ----------
        config : SpeakChatterboxTTSConfig
            Configuration for the connector.
        """
        super().__init__(config)

        # OM API key
        api_key = getattr(self.config, "api_key", None)

        # Sleep mode configuration
        self.io_provider = IOProvider()
        self.last_voice_command_time = time.time()

        # Chatterbox TTS configuration
        model_variant = self.config.model_variant
        device = self.config.device
        voice_prompt_path = self.config.voice_prompt_path
        exaggeration = self.config.exaggeration
        cfg_weight = self.config.cfg_weight
        language_id = self.config.language_id

        # silence rate
        self.silence_rate = self.config.silence_rate
        self.silence_counter = 0

        # IO Provider
        self.io_provider = IOProvider()

        self.audio_topic = "robot/status/audio"
        self.tts_status_request_topic = "om/tts/request"
        self.tts_status_response_topic = "om/tts/response"
        self.session = None
        self.audio_pub = None

        self.audio_status = AudioStatus(
            header=prepare_header(str(uuid4())),
            status_mic=AudioStatus.STATUS_MIC.UNKNOWN.value,
            status_speaker=AudioStatus.STATUS_SPEAKER.READY.value,
            sentence_to_speak=String(""),
        )

        try:
            self.session = open_zenoh_session()
            self.audio_pub = self.session.declare_publisher(self.audio_topic)
            self.session.declare_subscriber(self.audio_topic, self.zenoh_audio_message)
            self.session.declare_subscriber(
                self.tts_status_request_topic, self._zenoh_tts_status_request
            )
            self._zenoh_tts_status_response_pub = self.session.declare_publisher(
                self.tts_status_response_topic
            )

            if self.audio_pub:
                self.audio_pub.put(self.audio_status.serialize())

            logging.info("Chatterbox TTS Zenoh client opened")
        except Exception as e:
            logging.error(f"Error opening Chatterbox TTS Zenoh client: {e}")

        # Initialize Chatterbox TTS Provider
        self.tts = ChatterboxTTSProvider(
            model_variant=model_variant,
            device=device,
            voice_prompt_path=voice_prompt_path,
            exaggeration=exaggeration,
            cfg_weight=cfg_weight,
            language_id=language_id,
        )
        self.tts.start()

        # Configure Chatterbox TTS Provider to ensure settings are applied
        self.tts.configure(
            model_variant=model_variant,
            device=device,
            voice_prompt_path=voice_prompt_path,
            exaggeration=exaggeration,
            cfg_weight=cfg_weight,
            language_id=language_id,
        )

        # TTS status
        self.tts_enabled = True

        # Initialize conversation provider
        self.conversation_provider = TeleopsConversationProvider(api_key=api_key)

    def zenoh_audio_message(self, data: zenoh.Sample):
        """
        Process an incoming audio status message.

        Parameters
        ----------
        data : zenoh.Sample
            The Zenoh sample received, which should have a 'payload' attribute.
        """
        self.audio_status = AudioStatus.deserialize(data.payload.to_bytes())

    async def connect(self, output_interface: SpeakInput) -> None:
        """
        Process a speak action by sending text to Chatterbox TTS.

        Parameters
        ----------
        output_interface : SpeakInput
            The SpeakInput interface containing the text to be spoken.
        """
        if self.tts_enabled is False:
            logging.info("TTS is disabled, skipping TTS action")
            return

        if (
            self.silence_rate > 0
            and self.silence_counter < self.silence_rate
            and self.io_provider.llm_prompt is not None
            and "INPUT: Voice" not in self.io_provider.llm_prompt
        ):
            self.silence_counter += 1
            logging.info(
                f"Skipping TTS due to silence_rate {self.silence_rate}, counter {self.silence_counter}"
            )
            return

        self.silence_counter = 0

        # Add pending message to TTS
        pending_message = self.tts.create_pending_message(output_interface.action)

        # Store robot message to conversation history only if there was ASR input
        if (
            self.io_provider.llm_prompt is not None
            and "INPUT: Voice" in self.io_provider.llm_prompt
        ):
            self.conversation_provider.store_robot_message(output_interface.action)

        state = AudioStatus(
            header=prepare_header(str(uuid4())),
            status_mic=self.audio_status.status_mic,
            status_speaker=AudioStatus.STATUS_SPEAKER.ACTIVE.value,
            sentence_to_speak=String(json.dumps(pending_message)),
        )

        if self.audio_pub:
            self.audio_pub.put(state.serialize())
            return

        self.tts.add_pending_message(pending_message)

    def _zenoh_tts_status_request(self, data: zenoh.Sample):
        """
        Process an incoming TTS control status message.

        Parameters
        ----------
        data : zenoh.Sample
            The Zenoh sample received, which should have a 'payload' attribute.
        """
        tts_status = TTSStatusRequest.deserialize(data.payload.to_bytes())
        logging.debug(f"Received TTS Control Status message: {tts_status}")

        code = tts_status.code
        request_id = tts_status.request_id

        # Read the current status
        if code == 2:
            tts_status_response = TTSStatusResponse(
                header=prepare_header(tts_status.header.frame_id),
                request_id=request_id,
                code=1 if self.tts_enabled else 0,
                status=String(
                    data=("TTS Enabled" if self.tts_enabled else "TTS Disabled")
                ),
            )
            return self._zenoh_tts_status_response_pub.put(
                tts_status_response.serialize()
            )

        # Enable the TTS
        if code == 1:
            self.tts_enabled = True
            logging.debug("TTS Enabled")

            ai_status_response = TTSStatusResponse(
                header=prepare_header(tts_status.header.frame_id),
                request_id=request_id,
                code=1,
                status=String(data="TTS Enabled"),
            )
            return self._zenoh_tts_status_response_pub.put(
                ai_status_response.serialize()
            )

        # Disable the TTS
        if code == 0:
            self.tts_enabled = False
            logging.debug("TTS Disabled")
            ai_status_response = TTSStatusResponse(
                header=prepare_header(tts_status.header.frame_id),
                request_id=request_id,
                code=0,
                status=String(data="TTS Disabled"),
            )

            return self._zenoh_tts_status_response_pub.put(
                ai_status_response.serialize()
            )

    def stop(self) -> None:
        """Stop the Chatterbox TTS connector and cleanup resources."""
        if self.session:
            self.session.close()
            logging.info("Chatterbox TTS Zenoh client closed")

        if self.tts:
            self.tts.stop()
