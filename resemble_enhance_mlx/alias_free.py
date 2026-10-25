"""MLX port of the alias-free resamplers + SnakeBeta (BigVGAN AMP path).

The kaiser-sinc lowpass filter is fixed (no learned params), so we compute it
once and bake it as a constant, matching torch's kaiser_sinc_filter1d exactly.
"""
import math
import numpy as np
import mlx.core as mx
from .nn import convt_w_from_torch, conv_w_from_torch, replicate_pad1d


def kaiser_sinc_filter1d(cutoff, half_width, kernel_size):
    """Returns numpy filter of shape (1, 1, kernel_size). Mirrors torch impl."""
    even = (kernel_size % 2 == 0)
    half_size = kernel_size // 2
    delta_f = 4 * half_width
    A = 2.285 * (half_size - 1) * math.pi * delta_f + 7.95
    if A > 50.:
        beta = 0.1102 * (A - 8.7)
    elif A >= 21.:
        beta = 0.5842 * (A - 21) ** 0.4 + 0.07886 * (A - 21.)
    else:
        beta = 0.
    window = np.kaiser(kernel_size, beta).astype(np.float64)
    if even:
        time = np.arange(-half_size, half_size) + 0.5
    else:
        time = np.arange(kernel_size) - half_size
    if cutoff == 0:
        filt = np.zeros_like(time)
    else:
        filt = 2 * cutoff * window * np.sinc(2 * cutoff * time)  # np.sinc = sin(pi x)/(pi x)
        filt = filt / filt.sum()
    return filt.reshape(1, 1, kernel_size).astype(np.float32)


class LowPassFilter1d:
    def __init__(self, cutoff=0.5, half_width=0.6, stride=1, kernel_size=12):
        self.kernel_size = kernel_size
        even = (kernel_size % 2 == 0)
        self.pad_left = kernel_size // 2 - int(even)
        self.pad_right = kernel_size // 2
        self.stride = stride
        filt = kaiser_sinc_filter1d(cutoff, half_width, kernel_size)  # (1,1,ks)
        self.C = None
        self._filt_np = filt

    def _weight(self, C):
        # grouped conv1d: weight (C, 1, ks) -> mlx (C, ks, 1)
        w = np.broadcast_to(self._filt_np, (C, 1, self.kernel_size)).copy()
        return conv_w_from_torch(mx.array(w))

    def __call__(self, x):  # x: (N, L, C)
        C = x.shape[2]
        x = replicate_pad1d(x, self.pad_left, self.pad_right)
        return mx.conv1d(x, self._weight(C), stride=self.stride, groups=C)


class UpSample1d:
    def __init__(self, ratio=2, kernel_size=12):
        self.ratio = ratio
        self.kernel_size = kernel_size
        self.stride = ratio
        self.pad = kernel_size // ratio - 1
        self.pad_left = self.pad * self.stride + (kernel_size - self.stride) // 2
        self.pad_right = self.pad * self.stride + (kernel_size - self.stride + 1) // 2
        self._filt_np = kaiser_sinc_filter1d(0.5 / ratio, 0.6 / ratio, kernel_size)

    def _weight(self, C):
        w = np.broadcast_to(self._filt_np, (C, 1, self.kernel_size)).copy()  # (C,1,ks) torch convT layout
        return convt_w_from_torch(mx.array(w), groups=C)

    def __call__(self, x):  # x: (N, L, C)
        C = x.shape[2]
        x = replicate_pad1d(x, self.pad, self.pad)
        x = self.ratio * mx.conv_transpose1d(x, self._weight(C), stride=self.stride, groups=C)
        return x[:, self.pad_left:-self.pad_right, :]


class DownSample1d:
    def __init__(self, ratio=2, kernel_size=12):
        self.lowpass = LowPassFilter1d(cutoff=0.5 / ratio, half_width=0.6 / ratio,
                                       stride=ratio, kernel_size=kernel_size)

    def __call__(self, x):
        return self.lowpass(x)


class SnakeBeta:
    def __init__(self, log_alpha, log_beta, clamp=(1e-2, 50)):
        self.log_alpha = log_alpha  # (C,)
        self.log_beta = log_beta
        self.clamp = clamp

    def __call__(self, x):  # x: (N, L, C)
        alpha = mx.clip(mx.exp(self.log_alpha), self.clamp[0], self.clamp[1])
        beta = mx.clip(mx.exp(self.log_beta), self.clamp[0], self.clamp[1])
        return x + (1.0 / beta) * mx.square(mx.sin(x * alpha))


class UpActDown:
    def __init__(self, act, up_ratio=2, down_ratio=2, up_kernel_size=12, down_kernel_size=12):
        self.act = act
        self.upsample = UpSample1d(up_ratio, up_kernel_size)
        self.downsample = DownSample1d(down_ratio, down_kernel_size)

    def __call__(self, x):
        return self.downsample(self.act(self.upsample(x)))
