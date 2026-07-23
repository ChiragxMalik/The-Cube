#!/usr/bin/env python3
"""
test_tts.py — Standalone Piper TTS test.

Synthesizes and plays one or more sentences through the default audio output.

Usage:
    python tests/test_tts.py
    python tests/test_tts.py --text "Hello, I am Cube."
    python tests/test_tts.py --voice en_US-lessac-low
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import yaml  # noqa: E402

from tts import TTSSpeaker  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("test_tts")


def load_config() -> dict:
    cfg_path = Path(__file__).resolve().parent.parent / "config.yaml"
    if cfg_path.is_file():
        with open(cfg_path, "r") as f:
            return yaml.safe_load(f)
    return {}


def main() -> None:
    cfg = load_config()

    parser = argparse.ArgumentParser(description="Test Piper TTS synthesis + playback")
    parser.add_argument(
        "--text",
        default="Hello! I am Cube, your offline voice assistant. How can I help you today?",
    )
    parser.add_argument("--voice", default=cfg.get("tts_voice", "en_US-lessac-low"))
    args = parser.parse_args()

    logger.info("=== TTS Test (Piper) ===")
    logger.info("Voice: %s", args.voice)
    logger.info("Text:  '%s'\n", args.text)

    speaker = TTSSpeaker(voice=args.voice)

    # Test 1: Speak a single sentence
    logger.info("--- Test 1: Single sentence ---")
    t0 = time.monotonic()
    speaker.speak(args.text)
    elapsed = (time.monotonic() - t0) * 1000
    logger.info("Done in %.0f ms\n", elapsed)

    # Test 2: Speak multiple sentences via streaming
    logger.info("--- Test 2: Streaming multiple sentences ---")
    test_sentences = [
        "The quick brown fox jumps over the lazy dog.",
        "Pack my box with five dozen liquor jugs.",
        "How vexingly quick daft zebras jump!",
    ]

    def sentence_generator():
        for s in test_sentences:
            logger.info("  Yielding: '%s'", s)
            yield s

    t0 = time.monotonic()
    speaker.speak_stream(sentence_generator())
    elapsed = (time.monotonic() - t0) * 1000
    logger.info("Streaming test done — %d sentences in %.0f ms", len(test_sentences), elapsed)


if __name__ == "__main__":
    main()
