"""
cube.py — Main pipeline loop for the Cube offline voice assistant.

Pipeline flow:
  Mic (16 kHz mono, always capturing)
    → openWakeWord (always-on) detects wake word
    → [cooldown starts] begin recording command
    → Silero VAD endpoints when user stops speaking
    → whisper.cpp transcribes the command WAV
    → llama-server streams a response (non-thinking, sentence-split)
    → Piper speaks each sentence immediately (streaming TTS)
    → return to listening

Graceful shutdown on SIGTERM/SIGINT for systemd compatibility.
"""

from __future__ import annotations

import logging
import os
import signal
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import yaml

# ── Project-local imports ─────────────────────────────────────────────────────
from audio import AudioStream, make_temp_wav
from wake import WakeWordDetector
from vad import EndpointDetector
from stt import transcribe
from llm import LLMClient
from tts import TTSSpeaker

logger = logging.getLogger("cube")


# ── Configuration ─────────────────────────────────────────────────────────────

def load_config(path: str | Path = "config.yaml") -> dict[str, Any]:
    """Load and validate the YAML configuration file.

    Falls back to searching in the project root (one level up from ``src/``).
    """
    candidates = [
        Path(path),
        Path(__file__).resolve().parent.parent / "config.yaml",
    ]
    for candidate in candidates:
        if candidate.is_file():
            with open(candidate, "r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f)
            logger.info("Config loaded from %s", candidate)
            return cfg

    logger.error("config.yaml not found in %s", [str(c) for c in candidates])
    sys.exit(1)


# ── Pipeline ──────────────────────────────────────────────────────────────────

class CubePipeline:
    """Wires together all pipeline stages and runs the main loop."""

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self._running = False

        # ── Init components ───────────────────────────────────────────────
        sr = config.get("sample_rate", 16000)

        self.audio_stream = AudioStream(
            sample_rate=sr,
            channels=1,
            blocksize=1280,  # 80 ms @ 16 kHz — openWakeWord frame size
        )

        self.wake_detector = WakeWordDetector(
            model_name=config.get("wake_word_model", "hey_cube"),
            threshold=config.get("wake_threshold", 0.5),
            cooldown_seconds=config.get("cooldown_seconds", 2.0),
        )

        self.endpoint_detector = EndpointDetector(
            silence_ms=config.get("vad_silence_ms", 1000),
            max_record_seconds=config.get("max_record_seconds", 10.0),
            sample_rate=sr,
        )

        self.llm = LLMClient(
            server_url=config.get("llama_server_url", "http://127.0.0.1:8080"),
            system_prompt=config.get(
                "system_prompt",
                "Answer in 1-2 short spoken sentences, plainly. /no_think",
            ),
            max_tokens=config.get("max_tokens", 256),
            thinking_mode=config.get("thinking_mode", False),
            conversation_memory=config.get("conversation_memory", False),
            memory_turns=config.get("memory_turns", 4),
            context_limit=config.get("llama_context", 1024),
        )

        self.tts = TTSSpeaker(
            voice=config.get("tts_voice", "en_US-lessac-low"),
        )

        logger.info("CubePipeline initialised — all components ready.")

    # ── Main loop ─────────────────────────────────────────────────────────

    def run(self) -> None:
        """Start the always-on listening loop. Blocks until shutdown."""
        self._running = True

        with self.audio_stream:
            logger.info("🟢 Cube is listening…")
            while self._running:
                try:
                    chunk = self.audio_stream.read(timeout=2.0)
                except Exception:
                    # Timeout or transient error — just keep going
                    continue

                if self.wake_detector.detect(chunk):
                    self._handle_command()

        logger.info("Main loop exited.")

    def shutdown(self) -> None:
        """Signal the main loop to exit cleanly."""
        logger.info("Shutdown requested.")
        self._running = False

    # ── Command handler ───────────────────────────────────────────────────

    def _handle_command(self) -> None:
        """Record → transcribe → generate → speak."""
        pipeline_start = time.monotonic()

        # 1. Record until endpoint
        logger.info("📝 Recording command…")
        self.endpoint_detector.reset()
        self.audio_stream.flush()  # discard stale mic data

        while self._running:
            try:
                chunk = self.audio_stream.read(timeout=2.0)
            except Exception:
                continue
            if self.endpoint_detector.process(chunk):
                break

        recorded = self.endpoint_detector.get_recorded_audio()
        record_ms = (time.monotonic() - pipeline_start) * 1000
        logger.info("Recording done: %.1f s (%.0f ms)", len(recorded) / 16000, record_ms)

        if len(recorded) < 1600:  # < 0.1 s — probably a false trigger
            logger.warning("Recording too short (%.2f s), ignoring.", len(recorded) / 16000)
            self.wake_detector.reset()
            return

        # 2. Save WAV and transcribe
        t_stt = time.monotonic()
        wav_path = make_temp_wav(recorded, self.config.get("sample_rate", 16000))
        try:
            transcript = transcribe(
                wav_path=wav_path,
                whisper_binary=self.config.get(
                    "whisper_binary", "./whisper.cpp/build/bin/whisper-cli"
                ),
                model_path=self.config.get(
                    "whisper_model_path", "./whisper.cpp/models/ggml-tiny.en.bin"
                ),
            )
        finally:
            wav_path.unlink(missing_ok=True)

        stt_ms = (time.monotonic() - t_stt) * 1000
        logger.info("STT (%.0f ms): '%s'", stt_ms, transcript)

        if not transcript.strip():
            logger.warning("Empty transcript — skipping LLM/TTS.")
            self.wake_detector.reset()
            return

        # 3. Stream LLM → streaming TTS (sentence-by-sentence)
        t_llm = time.monotonic()
        sentence_stream = self.llm.stream_chat(transcript)
        self.tts.speak_stream(sentence_stream)
        llm_tts_ms = (time.monotonic() - t_llm) * 1000

        total_ms = (time.monotonic() - pipeline_start) * 1000
        logger.info(
            "Pipeline complete: record=%.0f ms  stt=%.0f ms  llm+tts=%.0f ms  total=%.0f ms",
            record_ms,
            stt_ms,
            llm_tts_ms,
            total_ms,
        )

        # Flush audio buffered during STT+LLM+TTS (several seconds of stale
        # data including Cube's own voice) before returning to the wake loop.
        self.audio_stream.flush()

        # Reset for next wake word
        self.wake_detector.reset()


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    """Load config, set up logging, wire signals, and run."""
    # Logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    config = load_config()
    pipeline = CubePipeline(config)

    # Graceful shutdown on SIGTERM / SIGINT
    def _signal_handler(signum: int, _frame: Any) -> None:
        sig_name = signal.Signals(signum).name
        logger.info("Received %s — shutting down…", sig_name)
        pipeline.shutdown()

    signal.signal(signal.SIGTERM, _signal_handler)
    signal.signal(signal.SIGINT, _signal_handler)

    try:
        pipeline.run()
    except KeyboardInterrupt:
        logger.info("KeyboardInterrupt — shutting down…")
        pipeline.shutdown()

    logger.info("Cube stopped.")


if __name__ == "__main__":
    main()
