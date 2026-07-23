#!/usr/bin/env python3
"""
test_wake.py — Standalone wake-word detection test.

Listens on the live microphone and prints when the wake word is detected.
Press Ctrl+C to stop.

Usage:
    python tests/test_wake.py                     # uses config.yaml defaults
    python tests/test_wake.py --model hey_jarvis  # explicit model
    python tests/test_wake.py --threshold 0.6     # custom threshold
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Add src/ to path so we can import project modules
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import yaml  # noqa: E402

from audio import AudioStream  # noqa: E402
from wake import WakeWordDetector  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("test_wake")


def load_config() -> dict:
    """Load config.yaml from project root."""
    cfg_path = Path(__file__).resolve().parent.parent / "config.yaml"
    if cfg_path.is_file():
        with open(cfg_path, "r") as f:
            return yaml.safe_load(f)
    return {}


def main() -> None:
    cfg = load_config()

    parser = argparse.ArgumentParser(description="Test wake-word detection from live mic")
    parser.add_argument("--model", default=cfg.get("wake_word_model", "hey_jarvis"))
    parser.add_argument("--threshold", type=float, default=cfg.get("wake_threshold", 0.5))
    parser.add_argument("--cooldown", type=float, default=cfg.get("cooldown_seconds", 2.0))
    args = parser.parse_args()

    logger.info("=== Wake Word Test ===")
    logger.info("Model: %s  |  Threshold: %.2f  |  Cooldown: %.1fs", args.model, args.threshold, args.cooldown)
    logger.info("Speak the wake word into your microphone. Press Ctrl+C to stop.\n")

    detector = WakeWordDetector(
        model_name=args.model,
        threshold=args.threshold,
        cooldown_seconds=args.cooldown,
    )

    sr = cfg.get("sample_rate", 16000)
    detection_count = 0

    try:
        with AudioStream(sample_rate=sr) as stream:
            while True:
                chunk = stream.read()
                if detector.detect(chunk):
                    detection_count += 1
                    logger.info("🔔 DETECTED #%d — wake word fired!", detection_count)
    except KeyboardInterrupt:
        logger.info("\nStopped. Total detections: %d", detection_count)


if __name__ == "__main__":
    main()
