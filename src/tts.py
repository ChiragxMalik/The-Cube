"""
tts.py — Text-to-speech via Piper, with streaming sentence playback.

Synthesizes one sentence at a time using ``piper-tts`` and plays it
immediately through the default audio output. This enables "streaming TTS":
as the LLM yields sentences, each is spoken before the full response is done.

TODO (v2): Barge-in / interrupting TTS when the user starts speaking.
           This needs acoustic echo cancellation (AEC) or the mic re-triggers
           on Cube's own voice. Mark clearly as v2.
"""

from __future__ import annotations

import io
import logging
import subprocess
import time
import wave
from typing import Generator, Optional

import numpy as np
import sounddevice as sd

logger = logging.getLogger(__name__)


class TTSSpeaker:
    """Synthesizes and plays text using Piper TTS.

    Args:
        voice: Piper voice shorthand (e.g. ``"en_US-lessac-low"``).
               Voice files are auto-downloaded on first use.
    """

    def __init__(self, voice: str = "en_US-lessac-low") -> None:
        self.voice = voice
        self._piper_available: Optional[bool] = None
        logger.info("TTSSpeaker ready — voice=%s", voice)

    # ── public API ────────────────────────────────────────────────────────

    def speak(self, text: str) -> None:
        """Synthesize *text* and play it through the speaker.

        Args:
            text: A single sentence or short phrase to speak.
        """
        if not text.strip():
            return

        logger.info("TTS speaking: '%s'", text[:80])
        t0 = time.monotonic()

        audio_data, sample_rate = self._synthesize(text)
        if audio_data is None:
            return

        synth_ms = (time.monotonic() - t0) * 1000
        logger.debug("TTS synthesis: %.0f ms (%.1f s audio)", synth_ms, len(audio_data) / sample_rate)

        # Play through default output device
        sd.play(audio_data, samplerate=sample_rate)
        sd.wait()

        total_ms = (time.monotonic() - t0) * 1000
        logger.debug("TTS total (synth+play): %.0f ms", total_ms)

    def speak_stream(self, sentence_gen: Generator[str, None, None]) -> None:
        """Speak sentences from a generator as they arrive.

        Each sentence is synthesized and played before the next one is
        requested, providing the lowest perceived latency.

        Args:
            sentence_gen: A generator yielding sentence strings.
        """
        logger.info("TTS streaming started")
        t0 = time.monotonic()
        count = 0

        for sentence in sentence_gen:
            self.speak(sentence)
            count += 1

        elapsed_ms = (time.monotonic() - t0) * 1000
        logger.info("TTS streaming done — %d sentences in %.0f ms", count, elapsed_ms)

    # ── internals ─────────────────────────────────────────────────────────

    def _synthesize(self, text: str) -> tuple[Optional[np.ndarray], int]:
        """Run Piper to synthesize *text* and return ``(audio_float32, sample_rate)``.

        Returns ``(None, 0)`` on failure.
        """
        try:
            result = subprocess.run(
                [
                    "piper",
                    "--model", self.voice,
                    "--output-raw",
                ],
                input=text.encode("utf-8"),   # must be bytes when text=False
                capture_output=True,
                text=False,                   # binary stdout for raw audio
                timeout=30,
            )
        except FileNotFoundError:
            # Try with python -m piper as fallback
            try:
                result = subprocess.run(
                    [
                        "python", "-m", "piper",
                        "--model", self.voice,
                        "--output-raw",
                    ],
                    input=text.encode("utf-8"),
                    capture_output=True,
                    timeout=30,
                )
            except FileNotFoundError:
                logger.error(
                    "Piper not found. Install with: pip install piper-tts"
                )
                return None, 0
        except subprocess.TimeoutExpired:
            logger.error("Piper timed out after 30 s")
            return None, 0

        if result.returncode != 0:
            stderr_msg = result.stderr.decode("utf-8", errors="replace") if isinstance(result.stderr, bytes) else result.stderr
            logger.error("Piper failed (rc=%d): %s", result.returncode, stderr_msg)
            return None, 0

        raw_audio = result.stdout
        if not raw_audio:
            logger.warning("Piper returned empty audio for: '%s'", text[:40])
            return None, 0

        # Piper --output-raw produces 16-bit signed LE PCM.
        # -low voices are 16 kHz; -medium/-high are 22050 Hz.
        # Switching voice without updating this causes wrong pitch/speed.
        if not self.voice.endswith("-low"):
            logger.warning(
                "Voice '%s' may not be 16 kHz. Only -low voices are tested. "
                "Use TTSSpeakerWav instead for correct sample-rate detection.",
                self.voice,
            )
        sample_rate = 16000
        audio_int16 = np.frombuffer(raw_audio, dtype=np.int16)
        audio_float32 = audio_int16.astype(np.float32) / 32768.0

        return audio_float32, sample_rate


class TTSSpeakerWav:
    """Alternative TTS speaker that writes to WAV files instead of --output-raw.

    Useful as a fallback if --output-raw behaves unexpectedly.
    """

    def __init__(self, voice: str = "en_US-lessac-low") -> None:
        self.voice = voice
        logger.info("TTSSpeakerWav ready — voice=%s", voice)

    def speak(self, text: str) -> None:
        """Synthesize *text* to a temp WAV and play it."""
        if not text.strip():
            return

        import tempfile
        from pathlib import Path

        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        tmp.close()
        tmp_path = tmp.name

        try:
            result = subprocess.run(
                [
                    "piper",
                    "--model", self.voice,
                    "--output_file", tmp_path,
                ],
                input=text,
                capture_output=True,
                text=True,
                timeout=30,
            )

            if result.returncode != 0:
                logger.error("Piper (WAV mode) failed: %s", result.stderr)
                return

            # Read and play the WAV
            with wave.open(tmp_path, "rb") as wf:
                sr = wf.getframerate()
                frames = wf.readframes(wf.getnframes())
                audio = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
                sd.play(audio, samplerate=sr)
                sd.wait()

        except FileNotFoundError:
            logger.error("Piper not found.")
        except subprocess.TimeoutExpired:
            logger.error("Piper (WAV mode) timed out")
        finally:
            Path(tmp_path).unlink(missing_ok=True)
