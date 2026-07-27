"""
vad.py — Voice Activity Detection for endpointing via Silero VAD.

Used ONLY for endpointing: after the wake word fires, Silero VAD detects
when the user stops speaking so the pipeline can hand the audio to STT.

Design note: VAD is NOT placed before the wake word — openWakeWord is
already always-on and cheap, and an aggressive VAD gate would clip the
start of the wake word.
"""

from __future__ import annotations

import logging
import time
from typing import Optional

import numpy as np
import torch

logger = logging.getLogger(__name__)


class EndpointDetector:
    """Accumulates audio after the wake word and detects end-of-speech.

    Args:
        silence_ms:        Consecutive silence (ms) that signals end-of-command.
        max_record_seconds: Hard cap on recording duration.
        sample_rate:        Audio sample rate (must be 16000 for Silero).
    """

    def __init__(
        self,
        silence_ms: int = 1000,
        max_record_seconds: float = 10.0,
        sample_rate: int = 16000,
    ) -> None:
        if sample_rate not in (8000, 16000):
            raise ValueError("Silero VAD only supports 8000 or 16000 Hz")

        self.silence_ms = silence_ms
        self.max_record_seconds = max_record_seconds
        self.sample_rate = sample_rate

        # Load Silero VAD model (cached locally after first download)
        self._model, _utils = torch.hub.load(
            repo_or_dir="snakers4/silero-vad",
            model="silero_vad",
            trust_repo=True,
        )
        logger.info(
            "EndpointDetector ready — silence_ms=%d  max_record=%.1fs",
            silence_ms,
            max_record_seconds,
        )

        # Silero VAD 5.x strictly requires exactly 512 samples at 16 kHz.
        # The mic blocksize (1280) is different, so we buffer incoming audio
        # and slice it into 512-sample windows before calling the model.
        self._vad_frame_size: int = 512
        self._vad_buffer: np.ndarray = np.array([], dtype=np.float32)

        # Recording state
        self._chunks: list[np.ndarray] = []
        self._recording_start: Optional[float] = None
        self._last_speech_time: Optional[float] = None
        self._speech_detected: bool = False

    # ── public API ────────────────────────────────────────────────────────

    def process(self, audio_chunk: np.ndarray) -> bool:
        """Feed one audio chunk and return ``True`` when end-of-speech is reached.

        Args:
            audio_chunk: 1-D float32 array, 16 kHz mono.

        Returns:
            ``True``  — user has stopped speaking (or hard cap reached).
            ``False`` — keep recording.
        """
        now = time.monotonic()

        if self._recording_start is None:
            self._recording_start = now
            self._last_speech_time = now  # assume speech starts immediately

        # Accumulate audio
        self._chunks.append(audio_chunk.copy())

        # Hard-cap check
        elapsed = now - self._recording_start
        if elapsed >= self.max_record_seconds:
            logger.info("EndpointDetector: hard cap reached (%.1f s)", elapsed)
            return True

        # Buffer incoming audio and process in strict 512-sample windows.
        # Silero VAD 5.x raises an error on any other chunk size at 16 kHz.
        self._vad_buffer = np.concatenate([self._vad_buffer, audio_chunk])
        while len(self._vad_buffer) >= self._vad_frame_size:
            frame = self._vad_buffer[: self._vad_frame_size]
            self._vad_buffer = self._vad_buffer[self._vad_frame_size :]
            tensor = torch.from_numpy(frame).float()
            speech_prob = self._model(tensor, self.sample_rate).item()
            if speech_prob >= 0.5:
                self._speech_detected = True
                self._last_speech_time = now

        # Only trigger endpoint after we've seen some speech
        if self._speech_detected and self._last_speech_time is not None:
            silence_duration_ms = (now - self._last_speech_time) * 1000
            if silence_duration_ms >= self.silence_ms:
                logger.info(
                    "EndpointDetector: endpoint after %.0f ms silence (total %.1f s)",
                    silence_duration_ms,
                    elapsed,
                )
                return True

        return False

    def get_recorded_audio(self) -> np.ndarray:
        """Return all accumulated audio as a single float32 array."""
        if not self._chunks:
            return np.array([], dtype=np.float32)
        return np.concatenate(self._chunks)

    def reset(self) -> None:
        """Clear accumulated audio and state for the next command."""
        self._chunks.clear()
        self._vad_buffer = np.array([], dtype=np.float32)
        self._recording_start = None
        self._last_speech_time = None
        self._speech_detected = False
        self._model.reset_states()
        logger.debug("EndpointDetector reset.")
