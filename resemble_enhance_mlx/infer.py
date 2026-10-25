"""Torch-free inference entry points: load the MLX package and enhance/denoise audio.

Mirrors resemble_enhance.inference: resample to 44.1 kHz, split into overlapping
chunks, run the model per chunk, and crossfade-merge. Resampling uses scipy
polyphase (kaiser) rather than torchaudio's sinc_interp_kaiser, so the resample
step is a close approximation, not bit-identical, for non-44.1k input.
"""
import math
import numpy as np
import mlx.core as mx
from scipy.signal import resample_poly
from . import serial

WAV_RATE = 44100


def load(path, lambd=0.5):
    """Load a resemble-enhance-mlx safetensors into a torch-free EnhancerM."""
    em = serial.load(path)
    em.lambd = lambd
    return em


def _resample(x, sr):
    if sr == WAV_RATE:
        return x
    g = math.gcd(WAV_RATE, sr)
    return resample_poly(x, WAV_RATE // g, sr // g, window=("kaiser", 14.77)).astype(np.float32)


def _merge(chunks, chunk_len, hop_len, length):
    sig_len = (len(chunks) - 1) * hop_len + chunk_len
    overlap = chunk_len - hop_len
    out = np.zeros(sig_len, dtype=np.float32)
    fadein = np.concatenate([np.linspace(0, 1, overlap, dtype=np.float32), np.ones(hop_len, np.float32)])
    fadeout = np.concatenate([np.ones(hop_len, np.float32), np.linspace(1, 0, overlap, dtype=np.float32)])
    for i, c in enumerate(chunks):
        if len(c) < chunk_len:
            c = np.pad(c, (0, chunk_len - len(c)))
        if i == 0:
            c = c * fadeout
        elif i == len(chunks) - 1:
            c = c * fadein
        else:
            c = c * fadein * fadeout
        s = i * hop_len
        out[s:s + chunk_len] += c[: len(out[s:s + chunk_len])]
    return out[:length]


def enhance(em, dwav, sr, chunk_seconds=30.0, overlap_seconds=1.0):
    """dwav: 1D np.float32. Returns (wav_44100, 44100)."""
    dwav = _resample(np.asarray(dwav, np.float32), sr)
    chunk_len = int(WAV_RATE * chunk_seconds)
    hop_len = chunk_len - int(WAV_RATE * overlap_seconds)
    outs = []
    for start in range(0, len(dwav), hop_len):
        seg = dwav[start:start + chunk_len]
        length = len(seg)
        am = np.abs(seg).max()
        am = am if am > 1e-7 else 1e-7
        seg = np.pad(seg / am, (0, 441))
        o = np.array(em(mx.array(seg)))[0][:length] * am
        outs.append(o)
        if start + chunk_len >= len(dwav):
            break
    hwav = _merge(outs, chunk_len, hop_len, length=len(dwav))
    return hwav, WAV_RATE
