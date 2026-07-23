"""
wake.py — Wake-word detection via openWakeWord.

Wraps ``openwakeword`` to detect a configurable wake word (default: "hey_jarvis").
A cooldown window prevents double-firing.

TODO (v2): Train a custom "Hey Cube" openWakeWord model and swap it in via config.
"""

from __future__ import annotations

import logging
import time

import numpy as np

logger = logging.getLogger(__name__)

# openWakeWord expects int16 @ 16 kHz
_OWW_SAMPLE_RATE = 16000


class WakeWordDetector:
    """Always-on, lightweight wake-word detector.

    Args:
        model_name: openWakeWord built-in model name (e.g. ``"hey_jarvis"``).
        threshold:  Detection confidence in ``[0, 1]``.
        cooldown_seconds: After a detection, suppress new detections for this
            many seconds to prevent double-firing.
    """

    def __init__(
        self,
        model_name: str = "hey_jarvis",
        threshold: float = 0.5,
        cooldown_seconds: float = 2.0,
    ) -> None:
        self.model_name = model_name
        self.threshold = threshold
        self.cooldown_seconds = cooldown_seconds
        self._last_trigger: float = 0.0

        # Lazy-load openwakeword so import cost is explicit
        import openwakeword  # noqa: F811
        from openwakeword.model import Model as OWWModel

        # Ensure built-in models are available
        openwakeword.utils.download_models()

        self._model = OWWModel(wakeword_models=[model_name])
        logger.info(
            "WakeWordDetector ready — model=%s  threshold=%.2f  cooldown=%.1fs",
            model_name,
            threshold,
            cooldown_seconds,
        )

    def detect(self, audio_chunk: np.ndarray) -> bool:
        """Process one audio chunk and return ``True`` if the wake word fires.

        Args:
            audio_chunk: 1-D float32 array, 16 kHz mono.

        Returns:
            ``True`` when the wake word is detected **and** the cooldown has
            elapsed since the last trigger.
        """
        # Cooldown check
        if time.monotonic() - self._last_trigger < self.cooldown_seconds:
            return False

        # openWakeWord expects int16 samples
        pcm16 = (audio_chunk * 32767).clip(-32768, 32767).astype(np.int16)
        prediction = self._model.predict(pcm16)

        score = prediction.get(self.model_name, 0.0)
        if score >= self.threshold:
            self._last_trigger = time.monotonic()
            logger.info("🔔 Wake word detected! (score=%.3f)", score)
            return True

        return False

    def reset(self) -> None:
        """Reset internal state (e.g. after a pipeline cycle completes)."""
        self._model.reset()
        logger.debug("WakeWordDetector state reset.")
