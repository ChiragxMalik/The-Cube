#!/usr/bin/env python3
"""
test_stt.py — Standalone whisper.cpp transcription test.

Transcribes a WAV file using the whisper.cpp binary specified in config.yaml.

Usage:
    python tests/test_stt.py                        # records from mic, then transcribes
    python tests/test_stt.py --wav /path/to/file.wav # transcribes an existing WAV
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np  # noqa: E402
import yaml  # noqa: E402

from audio import AudioStream, make_temp_wav  # noqa: E402
from stt import transcribe  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("test_stt")


def load_config() -> dict:
    cfg_path = Path(__file__).resolve().parent.parent / "config.yaml"
    if cfg_path.is_file():
        with open(cfg_path, "r") as f:
            return yaml.safe_load(f)
    return {}


def record_snippet(duration: float = 4.0, sample_rate: int = 16000) -> Path:
    """Record a short audio snippet from the microphone."""
    logger.info("Recording %.1f seconds from mic…", duration)
    chunks = []
    with AudioStream(sample_rate=sample_rate) as stream:
        t0 = time.monotonic()
        while time.monotonic() - t0 < duration:
            chunks.append(stream.read())
    audio = np.concatenate(chunks)
    path = make_temp_wav(audio, sample_rate)
    logger.info("Saved recording to %s (%.1f s)", path, len(audio) / sample_rate)
    return path


def main() -> None:
    cfg = load_config()

    parser = argparse.ArgumentParser(description="Test whisper.cpp transcription")
    parser.add_argument("--wav", type=str, default=None, help="Path to WAV file to transcribe")
    parser.add_argument("--duration", type=float, default=4.0, help="Mic recording duration if no WAV given")
    parser.add_argument("--whisper_binary", default=cfg.get("whisper_binary", "./whisper.cpp/build/bin/whisper-cli"))
    parser.add_argument("--model", default=cfg.get("whisper_model_path", "./whisper.cpp/models/ggml-tiny.en.bin"))
    args = parser.parse_args()

    sr = cfg.get("sample_rate", 16000)

    logger.info("=== STT Test (whisper.cpp) ===")
    logger.info("Binary: %s", args.whisper_binary)
    logger.info("Model:  %s\n", args.model)

    # Get WAV to transcribe
    temp_created = False
    if args.wav:
        wav_path = Path(args.wav)
        if not wav_path.is_file():
            logger.error("WAV file not found: %s", wav_path)
            sys.exit(1)
    else:
        logger.info("No --wav provided. Recording from mic for %.1f seconds…", args.duration)
        wav_path = record_snippet(args.duration, sr)
        temp_created = True

    # Transcribe
    t0 = time.monotonic()
    result = transcribe(
        wav_path=str(wav_path),
        whisper_binary=args.whisper_binary,
        model_path=args.model,
    )
    elapsed_ms = (time.monotonic() - t0) * 1000

    logger.info("─" * 60)
    logger.info("Transcript: '%s'", result)
    logger.info("Time:       %.0f ms", elapsed_ms)
    logger.info("─" * 60)

    # Cleanup temp file
    if temp_created:
        wav_path.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
