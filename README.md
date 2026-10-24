# AI Voice Studio 🎧 — phone voice note ➜ studio mic

**Turn your phone voice note into a studio-mic voice. Free, local, one command.**

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/OutOfAmine/ai-voice-studio/blob/main/ai_voice_studio.ipynb)
![License: MIT](https://img.shields.io/badge/license-MIT-green)
![Python 3.10](https://img.shields.io/badge/python-3.10-blue)
![GPU or CPU](https://img.shields.io/badge/runs%20on-GPU%20%7C%20CPU-orange)

Same take, same phone. Turn the sound on 🔊

https://github.com/user-attachments/assets/ad3ca9ae-6334-4b67-8e34-ae149464f1f7

No noise, no room echo, the treble lost to phone compression rebuilt, long pauses tightened, mixed to the loudness
Instagram and TikTok expect. Made by **Amine ([@justbyamine](https://www.instagram.com/justbyamine/))**: every
Darija tech reel on the account is voiced this way.

## Quick start

| You have | Do this |
|---|---|
| No GPU, nothing installed | Click **Open in Colab** above (or import [`ai_voice_studio.ipynb`](ai_voice_studio.ipynb) in Kaggle), Run all, open the link it prints |
| A PC (Windows/Linux) | Install (below), then `python app.py` and drop your voice note in the browser |
| A terminal habit | `python voice_pro.py voice-note.mp4 output` |

## Demo

**Before / after audio only:** [demo/before-after.mp3](demo/before-after.mp3)
(video above: [demo/before-after.mp4](demo/before-after.mp4), made with [`tools/make_demo_video.py`](tools/make_demo_video.py)
so you can build the same comparison for your own voice).

**The reel about this tool**, as published on TikTok (downloaded from TikTok Studio, with the @justbyamine
watermark). It plays the raw phone voice, then the same voice after the script; turn the sound on:

https://github.com/user-attachments/assets/dbd93934-ddee-4c4d-9f98-8867de06e567

Watch it on TikTok: https://www.tiktok.com/@justbyamine/video/7698607101580496180
(File in the repo: [demo/voice-pro-reel-from-tiktok.mp4](demo/voice-pro-reel-from-tiktok.mp4).)

## Proof: used on my published reels

- Reel about this tool, Instagram: https://www.instagram.com/reel/DetAeRqA1k8/
- Same reel on TikTok: https://www.tiktok.com/@justbyamine/video/7698607101580496180
- Same reel on Facebook: https://web.facebook.com/reel/1664386081973860/

Follow: [Instagram @justbyamine](https://www.instagram.com/justbyamine/) ·
[TikTok @justbyamine](https://www.tiktok.com/@justbyamine) ·
[Facebook: Just by amine](https://www.facebook.com/profile.php?id=61595322491388) ·
[GitHub @OutOfAmine](https://github.com/OutOfAmine)

## What it does

| Step | Output | What happens |
|---|---|---|
| 1. AI restoration | `1-restored.wav` | [resemble-enhance](https://github.com/resemble-ai/resemble-enhance) removes noise and room echo and rebuilds the treble lost to phone compression |
| 2. Pause tightening | `2-tight.wav` | pauses longer than 180 ms are shortened to 140 ms (8 ms crossfades) |
| 3. Broadcast chain | `3-chain.wav` | warmth (+3 dB @120 Hz), less "phone box" (−3 dB @380 Hz), presence (+2.5 dB @4.8 kHz), air, de-esser, 2-stage compression |
| 4. Final | `Voice-Pro-Final.wav` | quiet room tone at −66 dBFS (no dead digital silence), gain kept in float, peak limiter, then one measured linear gain to −14 LUFS with true peak ≤ −1.5 dBTP (Instagram / TikTok level, no pumping) |

It also writes `time-map.json` (old → new timings) so subtitles made on the original voice can be re-synced.
No pitch or speed change.

**Video in, video out:** give it a video (.mp4, .mov, .webm) and it also writes `Voice-Pro-Video.mp4`: the same
picture (not re-encoded) with the new voice. Pauses are kept for videos so the voice stays on the lips
(`VOICE_TIGHTEN=1` forces the tightening anyway; `VOICE_TIGHTEN=0` turns it off for audio too).

## Install (Windows, Python 3.10)

```
python -m venv .venv
.venv\Scripts\python -m pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121
.venv\Scripts\python -m pip install --no-deps resemble-enhance
.venv\Scripts\python -m pip install -r requirements.txt
```

No NVIDIA GPU? Install the normal torch (`pip install torch torchaudio`); it works on CPU, just slower.
The script stubs `deepspeed` (only needed for training) so resemble-enhance imports on Windows.

## Use

Web UI (upload or record, pick Strong / Soft, download the result):

On Windows, double-click **Launch Studio.cmd**. The app opens in your browser.
Keep the terminal open while using the studio; closing it stops the app.

- **1. Add your voice**: choose **I have a file** or **Record my voice**, add your audio, then click **Next**.
- **2. Clean up**: click **Make my voice better**. The recommended settings are already selected.
  Naming the recording, choosing Strong / Soft and turning off **Shorten long pauses** are under **Optional settings**.
- **3. Download**: play **Before** and **After** side by side, then click **Download my voice**. A video input
  also shows the video with the new voice, ready to download from its player.
- **Saved voices** keeps completed recordings, originals and processing results between app restarts.
  Files are stored in `local-library/` beside `app.py` and excluded from Git.
  Set `VOICE_STUDIO_LIBRARY` to choose another folder.

The default app listens only on `127.0.0.1`. Microphone access requires browser permission;
if the embedded browser has no microphone, open the local URL in Chrome or Edge.
The first enhancement may download model weights; subsequent processing uses local files.
`--share` remains an explicit opt-in for Colab: it exposes the app and library through a public link.

```
.venv\Scripts\python app.py
```

Command line:

```
.venv\Scripts\python voice_pro.py voice-note.mp4 output
```

Any audio or video input works (.mp4, .opus, .ogg, .m4a, .mp3, .wav). The result is `output/Voice-Pro-Final.wav`.
The first run downloads the models.

## Tips

- Record close to the phone, in a small room with soft things around (less echo).
- Cut your mistakes before running it; the script only shortens silences.
- Sounds "over-processed"? Run with a softer restoration: `set VOICE_LAMBD=0.5` (PowerShell: `$env:VOICE_LAMBD="0.5"`).
- Sounds hissy? Turn the 11 kHz air shelf down (it boosts highs the AI rebuilt): `set VOICE_AIR_DB=1` (default 3, 0 = off).
- Very soft syllables at the end of a phrase can be erased by the restoration: listen to the result, and re-run
  with `VOICE_LAMBD=0.5` if a word disappears.

---

### Darija

Script li kaydir l voice note dyal tilifoun soot b7al micro pro: kay7yed l'echo w l'souffle, kay9sser les silences
twal, kayzid chaleur w clarté, w kayjib l'volume dyal Instagram/TikTok. Kolchi f PC dyalek, fabor.

## Credits and license

[resemble-enhance](https://github.com/resemble-ai/resemble-enhance) (MIT) and [ffmpeg](https://ffmpeg.org/).
This repository: MIT, see [LICENSE](LICENSE).
