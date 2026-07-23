# The Cube — AI Assistant on a Raspberry Pi

Most "AI projects" are a thin wrapper around someone else's GPU in a datacenter. I
wanted more interesting version: getting a genuinely useful version running on small and cheap hardware.

A voice assistant that runs **entirely on a Raspberry Pi 4B (4 GB)**. No cloud, no
internet at runtime, no data leaving the device. You call it, it answers.

Everything the assistant needs — wake-word detection, speech recognition, the language
model, and text-to-speech — runs locally on the Pi. Models are downloaded once during
setup and cached on disk. Voices are swappable, and the LLM can be changed by pointing
one config line at a different `.gguf` file.

**The pipeline:** wake word → record your command → transcribe → LLM → speak the
answer → go back to listening.

---


**What it demonstrates**

- Running real speech + LLM inference on constrained ARM hardware, offline
- End-to-end embedded systems work: audio I/O, VAD, quantized models, systemd services
- Engineering judgment under real constraints — picking models by *measured* trade-offs, not vibes (see [Model Selection](#model-selection))
- Power/thermal awareness for a battery-powered device

---

## Table of Contents

- [Architecture](#architecture)
- [Prerequisites](#prerequisites)
- [Setup Guide](#setup-guide)
  - [1. Raspberry Pi OS](#1-raspberry-pi-os)
  - [2. System Packages](#2-system-packages)
  - [3. Python Virtual Environment](#3-python-virtual-environment)
  - [4. Build whisper.cpp & llama.cpp](#4-build-whispercpp--llamacpp)
  - [5. Download Models](#5-download-models)
  - [6. CPU Governor](#6-cpu-governor)
- [Model Selection](#model-selection)
- [Configuration](#configuration)
- [Testing Components](#testing-components)
- [Running the Full Pipeline](#running-the-full-pipeline)
- [systemd Services](#systemd-services)
- [v2 Roadmap](#v2-roadmap)
- [Troubleshooting](#troubleshooting)

---

## Architecture

```
Mic (always capturing 16 kHz mono)
  │
  ├── openWakeWord (always-on, cheap) ──► detects wake word
  │                                        │
  │                      [cooldown starts] │
  │                                        ▼
  ├── Silero VAD (endpointing) ──────────► stop recording when silence detected
  │                                        │
  │                                        ▼
  ├── whisper.cpp (tiny.en) ─────────────► transcribe command WAV
  │                                        │
  │                                        ▼
  ├── llama-server (Gemma 3 1B) ─────────► stream response tokens
  │        │                               │
  │        │  sentence boundary            ▼
  │        └──────────────────────► Piper TTS (speak each sentence immediately)
  │                                        │
  │                                        ▼
  └────────────────────────────── return to listening
```

The wake word runs constantly and is cheap. Once it fires, VAD figures out when you've
stopped talking so we don't wait on a fixed timer. TTS speaks sentence-by-sentence as
the LLM streams, so you hear the start of the answer before the model has finished
generating all of it — that's most of how the response *feels* fast.

### Project Structure

```
cube/
├── config.yaml              # All tunables (models, thresholds, paths)
├── requirements.txt         # Pinned Python dependencies
├── src/
│   ├── cube.py              # Main loop — wires the full pipeline
│   ├── audio.py             # Mic capture + playback helpers
│   ├── wake.py              # openWakeWord wrapper
│   ├── vad.py               # Silero VAD endpointing
│   ├── stt.py               # whisper.cpp subprocess wrapper
│   ├── llm.py               # llama-server streaming client
│   └── tts.py               # Piper streaming synthesis + playback
├── tests/
│   ├── test_wake.py         # Wake word from live mic
│   ├── test_vad.py          # Speech/silence segmentation
│   ├── test_stt.py          # Transcribe a sample WAV
│   ├── test_llm.py          # One prompt → streamed answer
│   └── test_tts.py          # Speak a sentence
├── systemd/
│   ├── llama-server.service # Runs llama-server with model
│   └── cube.service         # Runs the Cube pipeline
└── README.md
```

---

## Prerequisites

| Component       | Source                | Notes                                           |
|:----------------|:----------------------|:------------------------------------------------|
| Raspberry Pi 4B | 4 GB RAM              | ARM Cortex-A72                                  |
| OS              | Raspberry Pi OS Lite 64-bit | No desktop needed                          |
| whisper.cpp     | Built separately      | See below                                       |
| llama.cpp       | Built separately      | See below                                       |
| Python 3.11+    | System or pyenv       | PEP 668 — use a venv, not global pip            |
| Microphone      | USB or I2S            | ALSA-compatible                                 |
| Speaker         | 3.5 mm / USB / I2S    | ALSA-compatible                                 |

---

## Setup Guide

### 1. Raspberry Pi OS

Install **Raspberry Pi OS Lite 64-bit** with the Raspberry Pi Imager. Enable SSH and
set up Wi-Fi during imaging — you only need the internet for the initial setup, not at
runtime.

```bash
# After first boot, update:
sudo apt update && sudo apt upgrade -y
```

### 2. System Packages

```bash
sudo apt install -y \
    python3 python3-venv python3-pip \
    git cmake build-essential \
    libasound2-dev portaudio19-dev \
    alsa-utils
```

### 3. Python Virtual Environment

> ⚠️ Raspberry Pi OS is PEP 668 externally-managed — you **must** use a venv. Global
> `pip install` will just refuse.

```bash
cd ~/cube
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Build whisper.cpp & llama.cpp

The two C++ inference engines are built separately. This project's Python layer calls
the **prebuilt binaries** — it never tries to compile them itself.

**whisper.cpp:**
```bash
cd ~/whisper.cpp
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j4
# Binary: ~/whisper.cpp/build/bin/whisper-cli
```

**llama.cpp:**
```bash
cd ~/llama.cpp
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j4
# Binary: ~/llama.cpp/build/bin/llama-server
```

### 5. Download Models

> ⚠️ Always confirm the exact GGUF filename on each model's Hugging Face **Files** tab.
> Quant filenames drift slightly between repos and re-uploads.

**Speech-to-text — whisper tiny.en:**
```bash
cd ~/whisper.cpp/models
./download-ggml-model.sh tiny.en
# Produces: ggml-tiny.en.bin
```

**LLM — Gemma 3 1B (Q4_K_M) — the one I actually run:**
```bash
mkdir -p ~/models && cd ~/models
# Check: https://huggingface.co/bartowski/gemma-3-1b-it-GGUF/tree/main
wget https://huggingface.co/bartowski/gemma-3-1b-it-GGUF/resolve/main/gemma-3-1b-it-Q4_K_M.gguf
```

**LLM — Qwen3 0.6B (Q4_K_M) — solid lighter alternative:**
```bash
cd ~/models
# Check: https://huggingface.co/Qwen/Qwen3-0.6B-GGUF/tree/main
wget https://huggingface.co/Qwen/Qwen3-0.6B-GGUF/resolve/main/Qwen3-0.6B-Q4_K_M.gguf
```

See [Model Selection](#model-selection) for why I landed on Gemma 3 1B and how these two
compare in practice.

**Piper voice (auto-downloads):**
Piper voices download automatically the first time they're used, so there's no required
manual step. To pre-download:
```bash
piper --model en_US-lessac-low --download-dir ~/piper_voices < /dev/null
```

### 6. CPU Governor

The Pi 4 supports several CPU frequency governors. For a **battery build**, the default
`ondemand` / `schedutil` governor is the right call — it clocks up during inference and
back down when idle, which saves power and keeps the board cooler.

**Check current governor:**
```bash
cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor
```

**Set schedutil (recommended for battery):**
```bash
# Temporary (resets on reboot):
echo "schedutil" | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor
# Permanent: add that echo line to /etc/rc.local before "exit 0", or use a systemd unit.
```

**Set performance (mains-powered only):**
```bash
# Locks all cores at max clock — lowest latency, but wastes power and runs hot.
echo "performance" | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor
```

> 💡 I default to `schedutil`. The CPU still boosts to 1.5 GHz during inference — the
> governor just lets it drop back down when idle. `performance` is the plugged-in option
> if you want to squeeze out the last bit of latency.

---

## Model Selection

This was the part that took the most trial and error, so it's worth writing down.

I tried a spread of small GGUF models on the Pi 4 and judged them on one thing: does the
whole round trip (you finish talking → you hear the first words back) stay inside a
1–2 second budget while the answers are still actually useful? On a 4 GB Pi with no GPU,
those two goals fight each other constantly.

Here's what I found:

- **Bigger models (2B and up) were too slow.** They'd give nicer answers, but the
  latency blew right past what feels conversational. On this hardware, waiting several
  seconds for a reply kills the whole illusion.
- **The tiniest models were fast but too dumb.** Quick to respond, but they'd
  misunderstand simple questions or make things up often enough that I didn't trust
  them.
- **Qwen3 0.6B was a pleasant surprise** — genuinely quick and more capable than its
  size suggests. It's a great pick if you want to lean toward speed, and it's fully
  documented here as the alternative.
- **Gemma 3 1B (Q4_K_M) was the sweet spot for me.** A little more capable than the
  0.6B, still comfortably inside the latency budget, and no thinking-mode baggage. That's
  what the Cube runs by default.

**One thing I'd specifically steer you away from: thinking / reasoning models.** Even
when you disable thinking (e.g. Qwen3's `/no_think`), I found they still tended to run
slower and lean toward longer, more verbose outputs than a plain instruct model of the
same size — which is exactly the wrong trade when every hundred milliseconds is showing
up in the response time. For a device like this, a straightforward non-thinking instruct
model in the **0.6B–1B** range is the move.

If you want to run Qwen3 0.6B instead of Gemma, see
[Switching models](#switching-models) below.

---

## Configuration

All tunables live in [`config.yaml`](config.yaml). Key settings:

| Key                   | Default                         | Description                              |
|:----------------------|:--------------------------------|:-----------------------------------------|
| `wake_word_model`     | `hey_jarvis`                    | openWakeWord model name                  |
| `wake_threshold`      | `0.5`                           | Detection confidence threshold           |
| `cooldown_seconds`    | `2.0`                           | Ignore triggers for N seconds after fire |
| `vad_silence_ms`      | `1000`                          | Silence (ms) that ends recording         |
| `max_record_seconds`  | `10.0`                          | Hard recording cap                       |
| `whisper_binary`      | `./whisper.cpp/build/bin/whisper-cli` | Path to whisper-cli binary         |
| `whisper_model_path`  | `./whisper.cpp/models/ggml-tiny.en.bin` | Path to GGML model             |
| `llama_server_url`    | `http://127.0.0.1:8080`         | llama-server endpoint                    |
| `llama_context`       | `1024`                          | Context window size                      |
| `thinking_mode`       | `false`                         | Leave false. Only matters if you swap in a thinking-capable model like Qwen3 |
| `system_prompt`       | _"Answer in 1-2 short sentences…"_ | System message for the LLM            |
| `max_tokens`          | `256`                           | Max generated tokens                     |
| `conversation_memory` | `false`                         | Enable multi-turn memory                 |
| `memory_turns`        | `4`                             | Turn pairs to keep (if enabled)          |
| `tts_voice`           | `en_US-lessac-low`              | Piper voice model                        |
| `sample_rate`         | `16000`                         | Audio sample rate (Hz)                   |

### Switching models

The model isn't hard-coded — `llama-server` loads whatever `.gguf` you point it at, so
switching is a one-line change.

1. Edit `llama-server.service` (or your manual run command) to point `--model` at the
   other GGUF.
2. If you switch to **Qwen3 0.6B**, it's a thinking-capable model, so keep
   `thinking_mode: false` in `config.yaml` and make sure your prompt path sends
   `/no_think`. For **Gemma 3 1B** there's no thinking mode to worry about — the setting
   is simply ignored.

---

## Testing Components

Test each piece **standalone** before running the full pipeline — it makes debugging
enormously easier. Run everything from the project root with the venv active:

```bash
cd ~/cube
source venv/bin/activate
```

### 1. Wake word

Listens on the live mic and prints when the wake word fires.

```bash
python tests/test_wake.py
# Say "Hey Jarvis" into the mic — Ctrl+C to stop
python tests/test_wake.py --model hey_jarvis --threshold 0.6 --cooldown 2.0
```

### 2. VAD endpointing

Records from the mic and shows when speech starts and stops.

```bash
python tests/test_vad.py
python tests/test_vad.py --silence_ms 800 --max_seconds 15
python tests/test_vad.py --save /tmp/vad_test.wav   # save recorded audio
```

### 3. STT

Transcribes a WAV file or a quick mic recording.

```bash
python tests/test_stt.py                      # record ~4s from mic and transcribe
python tests/test_stt.py --wav ~/test_audio.wav
python tests/test_stt.py --whisper_binary ~/whisper.cpp/build/bin/whisper-cli \
                         --model ~/whisper.cpp/models/ggml-tiny.en.bin
```

### 4. LLM

Needs `llama-server` running first.

```bash
# Start llama-server (Gemma 3 1B):
~/llama.cpp/build/bin/llama-server \
    --model ~/models/gemma-3-1b-it-Q4_K_M.gguf \
    --ctx-size 1024 --threads 4 --host 127.0.0.1 --port 8080

# In another terminal:
python tests/test_llm.py
python tests/test_llm.py --prompt "What is the speed of light?"
```

### 5. TTS

Speaks a sentence through the speaker with Piper.

```bash
python tests/test_tts.py
python tests/test_tts.py --text "Hello, I am Cube."
python tests/test_tts.py --voice en_US-lessac-low
```

---

## Running the Full Pipeline

Once every component test passes:

```bash
# Terminal 1 — start llama-server (Gemma 3 1B by default):
~/llama.cpp/build/bin/llama-server \
    --model ~/models/gemma-3-1b-it-Q4_K_M.gguf \
    --ctx-size 1024 --threads 4 --host 127.0.0.1 --port 8080

# Terminal 2 — run Cube:
cd ~/cube
source venv/bin/activate
python src/cube.py
```

Say **"Hey Jarvis"**, ask your question, and the Cube answers and drops back into
listening. `Ctrl+C` sends SIGINT and shuts down gracefully.

> To run Qwen3 0.6B instead, swap the `--model` path for
> `~/models/Qwen3-0.6B-Q4_K_M.gguf`.

---

## systemd Services

For headless, boot-on-start operation:

```bash
sudo cp systemd/llama-server.service /etc/systemd/system/
sudo cp systemd/cube.service /etc/systemd/system/

# Edit paths if your layout isn't /home/pi/cube
sudo systemctl daemon-reload

sudo systemctl enable --now llama-server.service
sudo systemctl enable --now cube.service

sudo systemctl status llama-server
sudo systemctl status cube

journalctl -u cube -f
journalctl -u llama-server -f
```

`cube.service` depends on `llama-server.service`, so systemd brings the LLM up first.
Both restart automatically on failure.

---

## v2 Roadmap

Deliberately **out of scope for v1** — there are stubs and notes in the code for these.

| Feature                     | Notes                                                    |
|:----------------------------|:---------------------------------------------------------|
| 🔇 **Barge-in**             | Interrupt TTS when you start talking. Needs acoustic echo cancellation — without it, the mic re-triggers on the Cube's own voice. |
| 🔊 **Noise suppression**    | Auto-gain, noise gate, echo cancellation.                |
| 🗣️ **"Hey Cube" wake word** | Needs a custom-trained openWakeWord model. `hey_jarvis` is the stand-in for now. |

---

## Troubleshooting

### No audio input detected
```bash
arecord -l                                                 # list capture devices
arecord -D default -f S16_LE -r 16000 -c 1 -d 3 test.wav   # test recording
aplay test.wav                                             # play it back
```

### llama-server won't start
- Check free memory: `free -h` (need roughly 1.5 GB free for a 1B Q4_K_M model)
- Confirm the GGUF exists at the path in the service file
- Check logs: `journalctl -u llama-server -e`

### Piper TTS not found
```bash
pip install piper-tts
echo "Hello" | piper --model en_US-lessac-low --output-raw | aplay -r 16000 -f S16_LE
```

### whisper.cpp fails
- Binary built? `ls ~/whisper.cpp/build/bin/whisper-cli`
- Model downloaded? `ls ~/whisper.cpp/models/ggml-tiny.en.bin`
- Test manually: `~/whisper.cpp/build/bin/whisper-cli -m ~/whisper.cpp/models/ggml-tiny.en.bin -f test.wav`

### High latency
- Use the `schedutil` governor (or `performance` on mains power)
- Lower `max_tokens` in `config.yaml`
- Kill other heavy processes: `htop`
- If your build feels sluggish on Gemma 3 1B, try Qwen3 0.6B — it's lighter
- Avoid thinking/reasoning models (see [Model Selection](#model-selection))

---

**Built for the Raspberry Pi 4B · Fully offline · No cloud dependencies**