"""Offline torch -> MLX weight extraction (torch used here only, never at runtime).

Reads the EFFECTIVE weight from torch modules, so weight_norm parametrization is
already folded (torch exposes module.weight as the reconstructed g*v/||v||).
"""
import numpy as np
import mlx.core as mx
from .nn import (Conv1dM, ConvT1dM, Conv2dM, conv_w_from_torch,
                 convt_w_from_torch, conv2d_w_from_torch)


def _np(t):
    return mx.array(t.detach().cpu().numpy().astype(np.float32))


def conv1d_from_torch(tmod, stride=1, padding=0, dilation=1, groups=1, pad_mode="zeros"):
    w = conv_w_from_torch(_np(tmod.weight))
    b = _np(tmod.bias) if tmod.bias is not None else None
    return Conv1dM(w, b, stride, padding, dilation, groups, pad_mode)


def conv2d_from_torch(tmod, stride=1, padding=0, dilation=1, groups=1):
    w = conv2d_w_from_torch(_np(tmod.weight))
    b = _np(tmod.bias) if tmod.bias is not None else None
    return Conv2dM(w, b, stride, padding, dilation, groups)


def convt1d_from_torch(tmod, stride=1, padding=0, output_padding=0, groups=1):
    w = convt_w_from_torch(_np(tmod.weight), groups=groups)
    b = _np(tmod.bias) if tmod.bias is not None else None
    return ConvT1dM(w, b, stride, padding, output_padding, groups)
