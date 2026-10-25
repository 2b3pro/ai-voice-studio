# AI Voice Studio 🎧 (Mac Edition) — Apple Silicon Native

**Turn your phone voice note into a studio-mic voice. Free, 100% private, accelerated by Apple Silicon.**

[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)
![Version](https://img.shields.io/badge/version-v2.0.0--mac-blue?logo=apple)
![Apple Silicon](https://img.shields.io/badge/Apple%20Silicon-M1%20|%20M2%20|%20M3%20|%20M4%20|%20M5-black?logo=apple)
![macOS](https://img.shields.io/badge/macOS-13%2B-blue?logo=apple)
![Acceleration](https://img.shields.io/badge/accelerated%20by-Metal%20(MPS)-orange)
![Python 3.10--3.13](https://img.shields.io/badge/python-3.10%E2%80%933.13-blue)

Original audio restoration engine created by **Amine ([@justbyamine](https://www.instagram.com/justbyamine/))**. This fork is specifically ported and optimized as a **Mac-centric edition** designed for Apple Silicon hardware.

---

## Apple Silicon Features

* ⚡ **Metal Performance Shaders (MPS):** Full GPU acceleration on Apple Silicon unified memory for the [resemble-enhance](https://github.com/resemble-ai/resemble-enhance) continuous normalizing flow (CFM) diffusion model.
* 🎧 **Apple AudioToolbox (`aac_at`):** Uses macOS's native CoreAudio master-grade AAC encoder for high-fidelity audio muxing in exported videos.
* 🎬 **Apple VideoToolbox (`h264_videotoolbox`):** Offloads video muxing and transcoding to Apple Silicon's dedicated hardware Media Engine.
* 🎚️ **Interactive Air / Hiss Control:** Web UI slider under Optional Settings to dial in high-frequency boost (0 to 6 dB) and tame AI-generated treble/sibilance.
* 🚀 **Modern Python & NumPy 2.x Support:** Built-in runtime compatibility patches for NumPy 2.x and Python 3.12+ environments.
* 🖱️ **Finder One-Click Launcher:** Double-click [`Launch Studio.command`](Launch Studio.command) directly from macOS Finder.

---

## Quick Start (macOS)

### 1. Installation

Using [`uv`](https://github.com/astral-sh/uv) (recommended, fastest):

```bash
# Clone the repository
git clone https://github.com/your-username/ai-voice-studio.git
cd ai-voice-studio

# Create virtual environment
uv venv .venv --python 3.12
source .venv/bin/activate

# Install PyTorch with Apple Silicon MPS support
uv pip install torch torchaudio torchcodec

# Install resemblance enhancer & Mac audio dependencies
uv pip install --no-deps resemble-enhance
uv pip install -r requirements.txt
```

*(Alternatively, standard `python3 -m venv .venv` and `pip` work identically.)*

---

## How to Use

### Option A: Web UI (Recommended)

1. Double-click **`Launch Studio.command`** in macOS Finder (or run `python3 app.py` from terminal).
2. The studio opens automatically in your browser (`http://127.0.0.1:7860`).
3. **Step 1:** Upload your audio or video file (or record directly via microphone), then click **Continue**.
4. **Step 2:** Click **✨ Make my voice better**. *(Adjust **Air / Highs boost (dB)** under Optional Settings if your recording sounds hissy).*
5. **Step 3:** Preview **Before** and **After** side-by-side with interactive waveforms and download your enhanced audio or video.

### Option B: Terminal CLI

```bash
source .venv/bin/activate

# Basic usage
python3 voice_pro.py input_voice.m4a output_folder

# View all customization flags
python3 voice_pro.py --help

# Custom restoration strength & air shelf:
python3 voice_pro.py input_voice.m4a output_folder --strength natural --air 1.5

# High-speed draft (32 diffusion steps) without pause tightening:
python3 voice_pro.py interview.wav output_folder --nfe 32 --no-tighten

# Custom podcast loudness (-16 LUFS, -2.0 dBTP ceiling):
python3 voice_pro.py podcast.wav output_folder --air 0.0 --lufs -16.0 --tp -2.0
```

#### Available CLI Customization Flags

| Flag | Type / Choices | Default | Description |
|---|---|---|---|
| `-h`, `--help` | flag | — | Show CLI help message and exit |
| `-s`, `--strength` | `strong`, `natural` | `strong` | AI restoration profile (`strong` = `0.9`, `natural` = `0.5`) |
| `-l`, `--lambd` | float (`0.0`–`1.0`) | `0.9` | Custom denoising/enhancement weight (overrides `--strength`) |
| `-a`, `--air` | float (`0.0`–`6.0`) | `3.0` | High-frequency 11 kHz air shelf boost in dB |
| `--tighten` | flag | Auto | Shorten pauses longer than 180 ms to 140 ms (default on for audio) |
| `--no-tighten` | flag | Auto | Keep original timing (default on for video) |
| `-d`, `--device` | `auto`, `mps`, `cpu` | `auto` | Compute device (`mps` for Apple Silicon GPU, or `cpu`) |
| `--nfe` | int (`1`–`128`) | `64` | Continuous normalizing flow diffusion steps |
| `--solver` | `midpoint`, `rk4`, `euler` | `midpoint` | Numerical ODE solver |
| `--tau` | float (`0.0`–`1.0`) | `0.5` | Prior temperature for diffusion |
| `--lufs` | float | `-14.0` | Target broadcast loudness in LUFS |
| `--tp` | float | `-1.5` | Target maximum true peak in dBTP |

---

## What the Pipeline Does

| Step | Output | What happens |
|---|---|---|
| **1. AI restoration** | `1-restored.wav` | [resemble-enhance](https://github.com/resemble-ai/resemble-enhance) removes noise & room echo and rebuilds lost treble via Metal (MPS) |
| **2. Pause tightening** | `2-tight.wav` | Pauses longer than 180 ms are shortened to 140 ms with 8 ms crossfades (disabled on video by default to keep lipsync) |
| **3. Broadcast chain** | `3-chain.wav` | Proximity warmth (+3 dB @120 Hz), de-box (−3 dB @380 Hz), presence (+2.5 dB @4.8 kHz), adjustable air shelf, de-esser, and 2-stage compression |
| **4. Final mastering** | `Voice-Pro-Final.wav` | Subtle room tone at −66 dBFS, true-peak limiter, and measured linear gain to standard −14 LUFS (true peak ≤ −1.5 dBTP) with zero dynamic pumping |
| **5. Video muxing** | `Voice-Pro-Video.mp4` | Muxed with Apple **AudioToolbox** (`aac_at`) and hardware **VideoToolbox** (`h264_videotoolbox`) |

---

## Pro Tips for Best Results

- **Microphone technique:** Record close to your microphone/phone in a room with soft furnishings (rugs, curtains, couch) to minimize room reflection before processing.
- **Taming sibilance ("s" sounds):** If **Strong** mode sounds overly crisp or hissy, lower the **Air / Highs boost** slider to `0 dB` or `1 dB` (under Optional settings).
- **Softer cleanup:** If quiet end-of-sentence words get cut off, switch Cleanup mode to **Soft** (`VOICE_LAMBD=0.5`).
- **Subtitles:** The pipeline outputs `time-map.json` mapping original timestamps to tightened timestamps, allowing subtitle tracks to automatically realign.

---

## Credits & License

- Original AI Voice Studio architecture by [Amine (@justbyamine)](https://github.com/OutOfAmine).
- Enhancement model by [Resemble AI](https://github.com/resemble-ai/resemble-enhance) (MIT).
- FFmpeg audio/video processing via [FFmpeg](https://ffmpeg.org/).
- Licensed under the [MIT License](LICENSE).
