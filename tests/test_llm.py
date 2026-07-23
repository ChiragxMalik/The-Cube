#!/usr/bin/env python3
"""
test_llm.py — Standalone llama-server streaming test.

Sends a single prompt to the local llama-server and prints the streamed
response, sentence by sentence.

Requires llama-server to be running (e.g. via systemd or manually).

Usage:
    python tests/test_llm.py
    python tests/test_llm.py --prompt "What is the speed of light?"
    python tests/test_llm.py --url http://127.0.0.1:8080
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import yaml  # noqa: E402

from llm import LLMClient  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("test_llm")


def load_config() -> dict:
    cfg_path = Path(__file__).resolve().parent.parent / "config.yaml"
    if cfg_path.is_file():
        with open(cfg_path, "r") as f:
            return yaml.safe_load(f)
    return {}


def main() -> None:
    cfg = load_config()

    parser = argparse.ArgumentParser(description="Test llama-server streaming")
    parser.add_argument("--prompt", default="What is the capital of France?")
    parser.add_argument("--url", default=cfg.get("llama_server_url", "http://127.0.0.1:8080"))
    parser.add_argument("--max_tokens", type=int, default=cfg.get("max_tokens", 256))
    args = parser.parse_args()

    logger.info("=== LLM Streaming Test ===")
    logger.info("Server: %s", args.url)
    logger.info("Prompt: '%s'\n", args.prompt)

    client = LLMClient(
        server_url=args.url,
        system_prompt=cfg.get("system_prompt", "Answer in 1-2 short spoken sentences, plainly. /no_think"),
        max_tokens=args.max_tokens,
        thinking_mode=cfg.get("thinking_mode", False),
        conversation_memory=False,
    )

    t0 = time.monotonic()
    sentence_count = 0
    full_response = []

    logger.info("Streaming response:")
    logger.info("─" * 60)

    for sentence in client.stream_chat(args.prompt):
        sentence_count += 1
        elapsed = (time.monotonic() - t0) * 1000
        logger.info("[Sentence %d @ %.0f ms] %s", sentence_count, elapsed, sentence)
        full_response.append(sentence)

    total_ms = (time.monotonic() - t0) * 1000
    logger.info("─" * 60)
    logger.info("Full response: %s", " ".join(full_response))
    logger.info("Sentences: %d  |  Total time: %.0f ms", sentence_count, total_ms)


if __name__ == "__main__":
    main()
