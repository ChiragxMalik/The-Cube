"""
stt.py — Speech-to-text via whisper.cpp subprocess.

Calls the prebuilt ``whisper-cli`` binary on a temporary WAV file and
returns the transcription text. All paths are configurable.
"""

from __future__ import annotations

import logging
import subprocess
import time
from pathlib import Path

logger = logging.getLogger(__name__)


def transcribe(
    wav_path: str | Path,
    whisper_binary: str,
    model_path: str,
    language: str = "en",
) -> str:
    """Transcribe a WAV file using whisper.cpp.

    Args:
        wav_path:        Path to the 16 kHz mono WAV file to transcribe.
        whisper_binary:  Path to the compiled ``whisper-cli`` binary.
        model_path:      Path to the GGML model file (e.g. ``ggml-tiny.en.bin``).
        language:        Language code (default ``"en"``).

    Returns:
        The transcription as a stripped string, or ``""`` on failure.
    """
    wav_path = str(Path(wav_path).resolve())
    whisper_binary = str(Path(whisper_binary).resolve())
    model_path = str(Path(model_path).resolve())

    cmd = [
        whisper_binary,
        "-m", model_path,
        "-f", wav_path,
        "-l", language,
        "--no-timestamps",       # plain text output
        "-t", "4",               # threads (Pi 4 has 4 cores)
        # Note: --print-special is a bare toggle (no value). Default is off,
        # so we simply omit it rather than passing "false" as a positional arg.
    ]

    logger.info("STT command: %s", " ".join(cmd))
    t0 = time.monotonic()

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except FileNotFoundError:
        logger.error(
            "whisper-cli binary not found at '%s'. "
            "Make sure whisper.cpp is built and the path in config.yaml is correct.",
            whisper_binary,
        )
        return ""
    except subprocess.TimeoutExpired:
        logger.error("whisper.cpp timed out after 30 s")
        return ""

    elapsed_ms = (time.monotonic() - t0) * 1000

    if result.returncode != 0:
        logger.error("whisper.cpp failed (rc=%d):\n%s", result.returncode, result.stderr)
        return ""

    transcript = result.stdout.strip()
    # whisper.cpp may prefix lines with timestamps even with --no-timestamps;
    # strip anything that looks like "[HH:MM:SS.mmm --> ...]"
    cleaned_lines = []
    for line in transcript.splitlines():
        line = line.strip()
        if line.startswith("["):
            # Remove timestamp prefix
            bracket_end = line.find("]")
            if bracket_end != -1:
                line = line[bracket_end + 1:].strip()
        if line:
            cleaned_lines.append(line)

    transcript = " ".join(cleaned_lines)
    logger.info("STT result (%.0f ms): '%s'", elapsed_ms, transcript)
    return transcript
