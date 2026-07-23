#!/usr/bin/env python3
"""
test_vad.py — Standalone Silero VAD endpointing test.

Captures audio from the live microphone and prints real-time speech/silence
segmentation. Demonstrates how EndpointDetector decides when the user
stops speaking.

Press Ctrl+C to stop.

Usage:
    python tests/test_vad.py
    python tests/test_vad.py --silence_ms 800 --max_seconds 15
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import yaml  # noqa: E402

from audio import AudioStream, save_wav  # noqa: E402
from vad import EndpointDetector  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("test_vad")


def load_config() -> dict:
    cfg_path = Path(__file__).resolve().parent.parent / "config.yaml"
    if cfg_path.is_file():
        with open(cfg_path, "r") as f:
            return yaml.safe_load(f)
    return {}


def main() -> None:
    cfg = load_config()

    parser = argparse.ArgumentParser(description="Test Silero VAD endpointing from live mic")
    parser.add_argument("--silence_ms", type=int, default=cfg.get("vad_silence_ms", 1000))
    parser.add_argument("--max_seconds", type=float, default=cfg.get("max_record_seconds", 10.0))
    parser.add_argument("--save", type=str, default=None, help="Save recorded audio to this WAV path")
    args = parser.parse_args()

    sr = cfg.get("sample_rate", 16000)

    logger.info("=== VAD Endpointing Test ===")
    logger.info("Silence threshold: %d ms  |  Max recording: %.1f s", args.silence_ms, args.max_seconds)
    logger.info("Speak into your microphone. Recording will stop when you stop speaking.\n")

    detector = EndpointDetector(
        silence_ms=args.silence_ms,
        max_record_seconds=args.max_seconds,
        sample_rate=sr,
    )

    round_num = 0

    try:
        with AudioStream(sample_rate=sr) as stream:
            while True:
                round_num += 1
                logger.info("--- Round %d: Start speaking (or Ctrl+C to quit) ---", round_num)
                detector.reset()
                t0 = time.monotonic()

                # Wait for first speech-like chunk before starting endpointing
                while True:
                    chunk = stream.read()
                    if detector.process(chunk):
                        break

                elapsed = time.monotonic() - t0
                audio = detector.get_recorded_audio()
                logger.info(
                    "✅ Endpoint reached — %.2f s of audio captured (%d samples)",
                    elapsed,
                    len(audio),
                )

                if args.save:
                    out_path = Path(args.save).with_stem(f"{Path(args.save).stem}_{round_num}")
                    save_wav(audio, out_path, sr)
                    logger.info("Saved to %s", out_path)

                logger.info("")

    except KeyboardInterrupt:
        logger.info("\nStopped after %d rounds.", round_num)


if __name__ == "__main__":
    main()
