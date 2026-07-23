"""
audio.py — Microphone capture and playback helpers.

Provides:
  • AudioStream  — context-managed, always-on 16 kHz mono mic input.
  • play_raw()   — play raw PCM bytes through the default output device.
  • save_wav()   — persist a numpy float32 array to a WAV file.
"""

from __future__ import annotations

import logging
import queue
import tempfile
import wave
from pathlib import Path
from typing import Optional

import numpy as np
import sounddevice as sd

logger = logging.getLogger(__name__)


# ── Mic Capture ───────────────────────────────────────────────────────────────

class AudioStream:
    """Continuously captures 16 kHz mono audio in a background thread.

    Usage::

        with AudioStream(sample_rate=16000) as stream:
            while True:
                chunk = stream.read()   # np.ndarray float32, shape (blocksize,)
                ...
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        channels: int = 1,
        blocksize: int = 1280,       # 80 ms @ 16 kHz — matches openWakeWord frame
    ) -> None:
        self.sample_rate = sample_rate
        self.channels = channels
        self.blocksize = blocksize
        self._queue: queue.Queue[np.ndarray] = queue.Queue()
        self._stream: Optional[sd.InputStream] = None

    # -- context manager ----------------------------------------------------

    def __enter__(self) -> "AudioStream":
        self.start()
        return self

    def __exit__(self, *exc) -> None:  # noqa: ANN002
        self.stop()

    # -- public API ---------------------------------------------------------

    def start(self) -> None:
        """Open the mic stream and begin filling the internal queue."""
        logger.info(
            "Opening mic: %d Hz, %d ch, blocksize %d",
            self.sample_rate,
            self.channels,
            self.blocksize,
        )
        self._stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=self.channels,
            dtype="float32",
            blocksize=self.blocksize,
            callback=self._callback,
        )
        self._stream.start()

    def stop(self) -> None:
        """Close the mic stream gracefully."""
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None
            logger.info("Mic stream closed.")

    def read(self, timeout: float = 2.0) -> np.ndarray:
        """Return the next audio chunk (float32, mono, shape ``(blocksize,)``).

        Blocks until a chunk is available or *timeout* seconds elapse.

        Raises:
            queue.Empty: if no audio arrives within *timeout*.
        """
        chunk = self._queue.get(timeout=timeout)
        return chunk

    def flush(self) -> None:
        """Discard any buffered audio chunks."""
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
            except queue.Empty:
                break

    # -- internals ----------------------------------------------------------

    def _callback(
        self,
        indata: np.ndarray,
        frames: int,            # noqa: ARG002
        time_info: object,      # noqa: ARG002
        status: sd.CallbackFlags,
    ) -> None:
        if status:
            logger.warning("Sounddevice status: %s", status)
        # indata shape: (blocksize, channels) — squeeze to 1-D mono
        self._queue.put(indata[:, 0].copy())


# ── Playback ──────────────────────────────────────────────────────────────────

def play_raw(audio_bytes: bytes, sample_rate: int = 16000, width: int = 2) -> None:
    """Play raw PCM bytes (int16 LE) through the default output device.

    Args:
        audio_bytes: Raw signed 16-bit little-endian PCM data.
        sample_rate:  Playback sample rate.
        width:        Sample width in bytes (2 = int16).
    """
    audio_np = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
    sd.play(audio_np, samplerate=sample_rate)
    sd.wait()


def play_wav(wav_path: str | Path) -> None:
    """Play a WAV file through the default output device."""
    import soundfile as sf

    data, sr = sf.read(str(wav_path), dtype="float32")
    sd.play(data, samplerate=sr)
    sd.wait()


# ── WAV helpers ───────────────────────────────────────────────────────────────

def save_wav(
    audio: np.ndarray,
    path: str | Path,
    sample_rate: int = 16000,
) -> Path:
    """Write a float32 numpy array to a 16-bit PCM WAV file.

    Args:
        audio:       1-D float32 array (values in -1..1).
        path:        Destination file path.
        sample_rate: Sample rate of the audio.

    Returns:
        The resolved :class:`Path` that was written.
    """
    path = Path(path)
    pcm16 = (audio * 32767).clip(-32768, 32767).astype(np.int16)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm16.tobytes())
    logger.debug("Saved WAV: %s  (%d samples, %.2f s)", path, len(pcm16), len(pcm16) / sample_rate)
    return path


def make_temp_wav(audio: np.ndarray, sample_rate: int = 16000) -> Path:
    """Write audio to a temporary WAV file and return its path.

    The caller is responsible for deleting the file when done.
    """
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp.close()
    return save_wav(audio, tmp.name, sample_rate)
