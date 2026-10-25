"""MLX port of resemble-enhance's spectral denoiser (2D UNet + STFT masking).

UNet activations are carried channels-last (N, H=freq, W=time, C). The denoiser
STFT differs from the mel frontend: n_fft=win_length=1680, hop=420, hann window,
complex output with the last frame dropped.
"""
import math
import numpy as np
import mlx.core as mx
from .nn import group_norm
from .convert import conv2d_from_torch

HOP = 420
N_FFT = HOP * 4          # 1680
WIN = HOP * 4            # 1680
EPS = 1e-7


def gelu(x):  # exact erf GELU (torch nn.GELU default)
    return 0.5 * x * (1.0 + mx.erf(x / math.sqrt(2.0)))


def _hann(n):  # periodic hann, matches torch.hann_window(n)
    k = mx.arange(n)
    return 0.5 - 0.5 * mx.cos(2.0 * math.pi * k / n)


def stft(x):
    """x: (B, T) -> mag, cos, sin each (B, F=N_FFT//2+1, frames) (last frame dropped)."""
    B, T = x.shape
    p = N_FFT // 2
    # reflect pad (center=True, pad_mode='reflect')
    left = x[:, 1:p + 1][:, ::-1]
    right = x[:, -p - 1:-1][:, ::-1]
    xp = mx.concatenate([left, x, right], axis=1)            # (B, T+2p)
    n_frames = 1 + (xp.shape[1] - N_FFT) // HOP
    idx = mx.arange(N_FFT)[None, :] + (mx.arange(n_frames) * HOP)[:, None]
    frames = xp[:, idx] * _hann(WIN)                         # (B, n_frames, N_FFT)
    spec = mx.fft.rfft(frames, axis=-1)                      # (B, n_frames, F)
    spec = mx.transpose(spec, (0, 2, 1))                     # (B, F, n_frames)
    spec = spec[:, :, :-1]                                   # drop last frame
    mag = mx.abs(spec)
    re, im = spec.real, spec.imag
    # match torch angle().cos()/.sin(): when mag==0, angle=0 -> cos=1, sin=0
    cos = mx.where(mag > 0, re / mx.maximum(mag, EPS), mx.ones_like(mag))
    sin = mx.where(mag > 0, im / mx.maximum(mag, EPS), mx.zeros_like(mag))
    return mag, cos, sin


def istft(mag, cos, sin):
    """Inverse of stft. Inputs (B, F, t); returns (B, samples)."""
    real = mag * cos
    imag = mag * sin
    s = real + 1j * imag                                     # (B, F, t)
    # replicate pad (0,1) along time
    s = mx.concatenate([s, s[:, :, -1:]], axis=2)            # (B, F, t+1)
    B, F, T = s.shape
    frames = mx.fft.irfft(s, n=N_FFT, axis=1)                # (B, N_FFT, T)
    win = _hann(WIN)
    frames = frames * win[None, :, None]                     # windowed
    R = N_FFT // HOP                                          # overlap factor = 4
    chunks = frames.reshape(B, R, HOP, T)                    # split N_FFT into R blocks
    win_chunks = (win * win).reshape(R, HOP)
    nb = T + R - 1
    out = mx.zeros((B, HOP, nb))
    env = mx.zeros((HOP, nb))
    for r in range(R):
        out = out + mx.pad(chunks[:, r], [(0, 0), (0, 0), (r, R - 1 - r)])
        col = mx.broadcast_to(win_chunks[r][:, None], (HOP, T))
        env = env + mx.pad(col, [(0, 0), (r, R - 1 - r)])
    out = out / (env[None] + EPS)
    out = mx.transpose(out, (0, 2, 1)).reshape(B, nb * HOP)  # overlap-add -> (B, L)
    # center crop n_fft//2 each side
    return out[:, N_FFT // 2: -(N_FFT // 2)]


class _PreactResBlock:
    def __init__(self, t_block, dim):
        self.ng = dim // 16
        self.gn1_w, self.gn1_b = _gn(t_block[0])
        self.conv1 = conv2d_from_torch(t_block[2], padding=1)
        self.gn2_w, self.gn2_b = _gn(t_block[3])
        self.conv2 = conv2d_from_torch(t_block[5], padding=1)

    def __call__(self, x):
        h = group_norm(x, self.ng, self.gn1_w, self.gn1_b)
        h = self.conv1(gelu(h))
        h = group_norm(h, self.ng, self.gn2_w, self.gn2_b)
        h = self.conv2(gelu(h))
        return x + h


class _UNetBlock:
    def __init__(self, t_block, scale_factor):
        self.pre_conv = conv2d_from_torch(t_block.pre_conv, padding=1)
        self.res1 = _PreactResBlock(t_block.res_block1, t_block.res_block1[2].out_channels)
        self.res2 = _PreactResBlock(t_block.res_block2, t_block.res_block2[2].out_channels)
        self.scale = scale_factor

    def __call__(self, x, h=None):
        if self.scale > 1:          # nearest upsample on H,W (channels-last)
            x = mx.repeat(mx.repeat(x, 2, axis=1), 2, axis=2)
        if h is not None:
            x = x + h
        x = self.pre_conv(x)
        x = self.res1(x)
        x = self.res2(x)
        skip = x
        if self.scale < 1:          # nearest downsample by 0.5
            x = x[:, ::2, :, :][:, :, ::2, :]
        return x, skip


class UNetM:
    def __init__(self, t_net):
        self.input_proj = conv2d_from_torch(t_net.input_proj, padding=1)
        self.n_enc = len(t_net.encoder_blocks)
        self.scale = 2 ** self.n_enc
        self.encoder = [_UNetBlock(b, 0.5) for b in t_net.encoder_blocks]
        self.middle = [_UNetBlock(b, 1.0) for b in t_net.middle_blocks]
        self.decoder = [_UNetBlock(b, 2.0) for b in t_net.decoder_blocks]
        self.head_conv1 = conv2d_from_torch(t_net.head[0], padding=1)
        self.head_conv2 = conv2d_from_torch(t_net.head[2], padding=0)

    def _pad_to_fit(self, x):  # x: (N,H,W,C)
        H, W = x.shape[1], x.shape[2]
        hpad = (self.scale - H % self.scale) % self.scale
        wpad = (self.scale - W % self.scale) % self.scale
        return mx.pad(x, [(0, 0), (0, hpad), (0, wpad), (0, 0)]), H, W

    def __call__(self, x):  # x: (N,H,W,C)
        x, H, W = self._pad_to_fit(x)
        x = self.input_proj(x)
        skips = []
        for blk in self.encoder:
            x, s = blk(x)
            skips.append(s)
        for blk in self.middle:
            x, _ = blk(x)
        for blk, s in zip(self.decoder, reversed(skips)):
            x, _ = blk(x, s)
        x = self.head_conv2(gelu(self.head_conv1(x)))
        return x[:, :H, :W, :]


class DenoiserM:
    def __init__(self, t_den):
        self.net = UNetM(t_den.net)

    def __call__(self, x):  # x: (B, T) waveform
        x = _normalize(x)
        mag, cos, sin = stft(x)                              # (B, F, t)
        mag_mask, sin_res, cos_res = self._predict(mag, cos, sin)
        sep_mag = mx.maximum(mag * mag_mask, 0.0)
        sep_cos = cos * cos_res - sin * sin_res
        sep_sin = sin * cos_res + cos * sin_res
        o = istft(sep_mag, sep_cos, sep_sin)
        npad = x.shape[-1] - o.shape[-1]
        if npad > 0:
            o = mx.pad(o, [(0, 0), (0, npad)])
        elif npad < 0:
            o = o[:, :x.shape[-1]]
        return o

    def _predict(self, mag, cos, sin):
        # stack as channels-last: (B, F, t, 3)
        x = mx.stack([mag, cos, sin], axis=-1)
        y = self.net(x)                                      # (B, F, t, 3)
        mag_mask = mx.sigmoid(y[..., 0])
        real = mx.tanh(y[..., 1])
        imag = mx.tanh(y[..., 2])
        m = mx.sqrt(real * real + imag * imag + EPS)
        cos_res = real / m
        sin_res = imag / m
        return mag_mask, sin_res, cos_res


def _normalize(x):
    return x / (mx.max(mx.abs(x), axis=-1, keepdims=True) + EPS)


def _gn(t_gn):
    return (mx.array(np.asarray(t_gn.weight.detach().cpu().numpy(), np.float32)),
            mx.array(np.asarray(t_gn.bias.detach().cpu().numpy(), np.float32)))
