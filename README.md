# AI Voice Studio 🎧 — WhatsApp voice note ➜ studio mic

Turn a phone / WhatsApp voice note into a voice that sounds like it was recorded on a pro microphone.
Local, free, no subscription. Made by **Amine ([@justbyamine](https://www.instagram.com/justbyamine/))** for his
Darija tech reels: every reel on the account is voiced this way.

## Demo

**Before / after** (raw WhatsApp voice, short pause, then the same take after the script):
[demo/before-after.mp3](demo/before-after.mp3)

**The reel about this tool**, as published on TikTok (downloaded from TikTok Studio, with the @justbyamine
watermark). It plays the raw WhatsApp voice, then the same voice after the script; turn the sound on:

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
| 1. AI restoration | `1-restored.wav` | [resemble-enhance](https://github.com/resemble-ai/resemble-enhance) removes noise and room echo and rebuilds the treble WhatsApp cuts off |
| 2. Pause tightening | `2-tight.wav` | pauses longer than 180 ms are shortened to 140 ms (8 ms crossfades) |
| 3. Broadcast chain | `3-chain.wav` | warmth (+3 dB @120 Hz), less "phone box" (−3 dB @380 Hz), presence (+2.5 dB @4.8 kHz), air, de-esser, 2-stage compression |
| 4. Final | `Voice-Pro-Final.wav` | quiet room tone at −66 dBFS (no dead digital silence), loudness −14 LUFS / −1.5 dBTP (Instagram / TikTok level) |

It also writes `time-map.json` (old → new timings) so subtitles made on the original voice can be re-synced.
No pitch or speed change.

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

```
.venv\Scripts\python voice_pro.py "WhatsApp Audio.mp4" output
```

Any audio or video input works (.mp4, .opus, .ogg, .m4a, .mp3, .wav). The result is `output/Voice-Pro-Final.wav`.
The first run downloads the models.

## Tips

- Record close to the phone, in a small room with soft things around (less echo).
- Cut your mistakes before running it; the script only shortens silences.
- Sounds "over-processed"? Run with a softer restoration: `set VOICE_LAMBD=0.5` (PowerShell: `$env:VOICE_LAMBD="0.5"`).
- Very soft syllables at the end of a phrase can be erased by the restoration: listen to the result, and re-run
  with `VOICE_LAMBD=0.5` if a word disappears.

---

### Darija

Script li kaydir l voice note dyal WhatsApp soot b7al micro pro: kay7yed l'echo w l'souffle, kay9sser les silences
twal, kayzid chaleur w clarté, w kayjib l'volume dyal Instagram/TikTok. Kolchi f PC dyalek, fabor.

## Credits and license

[resemble-enhance](https://github.com/resemble-ai/resemble-enhance) (MIT) and [ffmpeg](https://ffmpeg.org/).
This repository: MIT, see [LICENSE](LICENSE).
