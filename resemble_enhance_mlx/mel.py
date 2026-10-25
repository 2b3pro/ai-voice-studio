"""MLX port of resemble-enhance's mel frontend.

Parity with resemble_enhance.melspec.MelSpectrogram is achieved by baking
torchaudio's slaney mel filterbank and Hann window as constants (extracted
once via tests/extract_mel_reference.py) rather than re-deriving slaney math.
"""
import numpy as np
import mlx.core as mx

# resemble-enhance HParams (base)
N_FFT = 2048
HOP = 420
WIN = 2048
NUM_MELS = 128
PREEMPHASIS = 0.97
STFT_MAGNITUDE_MIN = 1e-4
MIN_LEVEL_DB = -80.0  # 20*log10(1e-4)
HEADROOM_DB = 15.0


class MelMLX:
    def __init__(self, fb_path: str, window_path: str):
        # fb: (n_freqs=1025, n_mels=128); window: (2048,)
        self.fb = mx.array(np.load(fb_path).astype(np.float32))
        self.window = mx.array(np.load(window_path).astype(np.float32))
        self.pad = N_FFT // 2  # center=True

    def __call__(self, wav: mx.array) -> mx.array:
        """wav: (T,) mono float -> mel: (NUM_MELS, n_frames)."""
        # preemphasis: y[i] = x[i] - 0.97 * x[i-1], x[-1]=0
        shifted = mx.concatenate([mx.zeros((1,)), wav[:-1]])
        wav = wav - PREEMPHASIS * shifted

        # center padding (constant/zeros)
        wav = mx.pad(wav, (self.pad, self.pad))

        n_frames = 1 + (wav.shape[0] - N_FFT) // HOP
        idx = mx.arange(N_FFT)[None, :] + (mx.arange(n_frames) * HOP)[:, None]
        frames = wav[idx] * self.window  # (n_frames, N_FFT)

        spec = mx.fft.rfft(frames, axis=-1)        # (n_frames, 1025) complex
        mag = mx.abs(spec)                         # power=1 -> magnitude
        mel = mag @ self.fb                        # (n_frames, NUM_MELS)
        mel = mel.T                                # (NUM_MELS, n_frames)

        # amp_to_db + normalize
        mel = mx.log10(mx.maximum(mel, STFT_MAGNITUDE_MIN)) * 20.0
        mel = (mel - MIN_LEVEL_DB) / (-MIN_LEVEL_DB + HEADROOM_DB)
        return mel
