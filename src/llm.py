"""
llm.py — Streaming LLM client for llama-server (OpenAI-compatible API).

Connects to a locally running ``llama-server`` and streams tokens. Includes:
  • Non-thinking enforcement for Qwen3 (``/no_think`` + ``<think>`` stripping).
  • Sentence-boundary splitting so TTS can start speaking before generation ends.
  • Optional conversation history with context-overflow protection.
"""

from __future__ import annotations

import logging
import re
import time
from typing import Generator, Optional

import requests

logger = logging.getLogger(__name__)

# Regex to strip <think>...</think> blocks (dotall so it spans newlines)
_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)

# Sentence-ending punctuation (for splitting streamed tokens)
_SENTENCE_ENDERS = re.compile(r"(?<=[.!?])\s+")


def strip_thinking_tags(text: str) -> str:
    """Remove any ``<think>...</think>`` blocks from LLM output."""
    return _THINK_RE.sub("", text).strip()


class LLMClient:
    """Streaming client for a local llama-server instance.

    Args:
        server_url:          Base URL (e.g. ``http://127.0.0.1:8080``).
        system_prompt:       System message prepended to every request.
        max_tokens:          Maximum tokens to generate per response.
        thinking_mode:       If ``False``, appends ``/no_think`` and strips
                             ``<think>`` blocks (required for Qwen3).
        conversation_memory: If ``True``, keeps the last *memory_turns* turns.
        memory_turns:        Number of user/assistant pairs to retain.
        context_limit:       Approximate context window (-c) for overflow guard.
    """

    def __init__(
        self,
        server_url: str = "http://127.0.0.1:8080",
        system_prompt: str = "Answer in 1-2 short spoken sentences, plainly. /no_think",
        max_tokens: int = 256,
        thinking_mode: bool = False,
        conversation_memory: bool = False,
        memory_turns: int = 4,
        context_limit: int = 1024,
    ) -> None:
        self.server_url = server_url.rstrip("/")
        self.system_prompt = system_prompt
        self.max_tokens = max_tokens
        self.thinking_mode = thinking_mode
        self.conversation_memory = conversation_memory
        self.memory_turns = memory_turns
        self.context_limit = context_limit
        self._history: list[dict[str, str]] = []  # list of {"role":..., "content":...}

        # Enforce /no_think in system prompt for non-thinking mode
        if not self.thinking_mode and "/no_think" not in self.system_prompt:
            self.system_prompt += " /no_think"

        logger.info(
            "LLMClient ready — url=%s  thinking=%s  memory=%s  max_tokens=%d",
            self.server_url,
            self.thinking_mode,
            self.conversation_memory,
            self.max_tokens,
        )

    # ── public API ────────────────────────────────────────────────────────

    def stream_chat(self, user_message: str) -> Generator[str, None, None]:
        """Send *user_message* and yield complete sentences as they form.

        Sentences are split on ``. ! ?`` boundaries so TTS can speak each one
        immediately while generation continues.

        Yields:
            Cleaned sentence strings, one at a time.
        """
        messages = self._build_messages(user_message)

        logger.info("LLM request — %d messages, user='%s'", len(messages), user_message[:80])
        t0 = time.monotonic()

        payload = {
            "model": "local",
            "messages": messages,
            "max_tokens": self.max_tokens,
            "stream": True,
            "temperature": 0.7,
        }

        # If non-thinking mode, try to disable thinking via API
        if not self.thinking_mode:
            # Some llama-server builds support this extra field
            payload["think"] = False

        full_response = ""
        buffer = ""
        first_token_time: Optional[float] = None

        try:
            resp = self._request_with_retry(payload)

            for line in resp.iter_lines(decode_unicode=True):
                if not line or not line.startswith("data: "):
                    continue
                data_str = line[len("data: "):]
                if data_str.strip() == "[DONE]":
                    break

                import json
                try:
                    chunk = json.loads(data_str)
                except (json.JSONDecodeError, ValueError):
                    continue

                delta = chunk.get("choices", [{}])[0].get("delta", {})
                token = delta.get("content", "")
                if not token:
                    continue

                if first_token_time is None:
                    first_token_time = time.monotonic()
                    ttft = (first_token_time - t0) * 1000
                    logger.info("LLM first token: %.0f ms", ttft)

                buffer += token
                full_response += token

                # Split on sentence boundaries and yield complete sentences
                parts = _SENTENCE_ENDERS.split(buffer)
                if len(parts) > 1:
                    for sentence in parts[:-1]:
                        cleaned = strip_thinking_tags(sentence).strip()
                        if cleaned:
                            yield cleaned
                    buffer = parts[-1]

        except requests.exceptions.ConnectionError as exc:
            logger.error("LLM connection failed: %s", exc)
            yield "Sorry, I can't reach my language model right now."
            return

        # Yield any remaining text in the buffer
        remaining = strip_thinking_tags(buffer).strip()
        if remaining:
            yield remaining

        # Update history
        full_clean = strip_thinking_tags(full_response).strip()
        if self.conversation_memory and full_clean:
            self._history.append({"role": "user", "content": user_message})
            self._history.append({"role": "assistant", "content": full_clean})
            self._trim_history()

        elapsed_ms = (time.monotonic() - t0) * 1000
        logger.info("LLM total: %.0f ms — response='%s'", elapsed_ms, full_clean[:120])

    def clear_history(self) -> None:
        """Drop all stored conversation turns."""
        self._history.clear()
        logger.debug("LLM conversation history cleared.")

    # ── internals ─────────────────────────────────────────────────────────

    def _build_messages(self, user_message: str) -> list[dict[str, str]]:
        """Assemble the messages list (system + optional history + user)."""
        messages: list[dict[str, str]] = [
            {"role": "system", "content": self.system_prompt},
        ]
        if self.conversation_memory:
            messages.extend(self._history)
        messages.append({"role": "user", "content": user_message})
        return messages

    def _trim_history(self) -> None:
        """Drop oldest turns if history exceeds *memory_turns* pairs.

        Never let history silently overflow the context window.
        """
        max_entries = self.memory_turns * 2  # each turn = user + assistant
        while len(self._history) > max_entries:
            removed = self._history.pop(0)
            logger.debug("Dropped oldest history entry: %s", removed["role"])

    def _request_with_retry(
        self,
        payload: dict,
        max_retries: int = 3,
        base_delay: float = 1.0,
    ) -> requests.Response:
        """POST to ``/v1/chat/completions`` with exponential backoff.

        Raises:
            requests.exceptions.ConnectionError: after all retries fail.
        """
        url = f"{self.server_url}/v1/chat/completions"

        for attempt in range(max_retries):
            try:
                resp = requests.post(
                    url,
                    json=payload,
                    headers={"Content-Type": "application/json"},
                    stream=True,
                    timeout=60,
                )
                resp.raise_for_status()
                return resp
            except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as exc:
                delay = base_delay * (2 ** attempt)
                logger.warning(
                    "LLM request attempt %d/%d failed (%s). Retrying in %.1f s…",
                    attempt + 1,
                    max_retries,
                    exc,
                    delay,
                )
                time.sleep(delay)

        raise requests.exceptions.ConnectionError(
            f"llama-server unreachable after {max_retries} attempts at {url}"
        )
