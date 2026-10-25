"""MLX conv primitives mirroring torch semantics, plus non-zero paddings.

Convention: our MLX modules carry activations channels-last (N, L, C).
Torch weights are converted at load time:
  Conv1d:          (C_out, C_in/g, K) -> (C_out, K, C_in/g)
  ConvTranspose1d: (C_in, C_out/g, K) -> (C_out/g, K, C_in)   [validated empirically]
"""
import mlx.core as mx


def conv_w_from_torch(w):  # (C_out, C_in/g, K) -> (C_out, K, C_in/g)
    return mx.transpose(w, (0, 2, 1))


def conv2d_w_from_torch(w):  # (C_out, C_in/g, KH, KW) -> (C_out, KH, KW, C_in/g)
    return mx.transpose(w, (0, 2, 3, 1))


def convt_w_from_torch(w, groups=1):
    # torch ConvTranspose1d weight: (C_in, C_out/g, K)
    # MLX conv_transpose1d weight: (C_out, K, C_in/g)
    Cin, Cog, K = w.shape
    Cout = Cog * groups
    if groups == 1:
        return mx.transpose(w, (1, 2, 0))  # (C_out, K, C_in)
    # grouped: reshape to expose groups on the C_in axis
    w = mx.reshape(w, (groups, Cin // groups, Cog, K))   # (g, Cin/g, Cout/g, K)
    w = mx.transpose(w, (0, 2, 3, 1))                    # (g, Cout/g, K, Cin/g)
    return mx.reshape(w, (Cout, K, Cin // groups))


class Conv1dM:
    """Runtime Conv1d on channels-last (N,L,C). Weight already in MLX layout."""
    def __init__(self, w, b=None, stride=1, padding=0, dilation=1, groups=1, pad_mode="zeros"):
        self.w, self.b = w, b
        self.stride, self.dilation, self.groups = stride, dilation, groups
        self.padding, self.pad_mode = padding, pad_mode

    def __call__(self, x):
        if self.pad_mode == "reflect":
            x = reflect_pad1d(x, self.padding)
            pad = 0
        else:
            pad = self.padding
        y = mx.conv1d(x, self.w, stride=self.stride, padding=pad,
                      dilation=self.dilation, groups=self.groups)
        if self.b is not None:
            y = y + self.b
        return y


class ConvT1dM:
    def __init__(self, w, b=None, stride=1, padding=0, output_padding=0, groups=1):
        self.w, self.b = w, b
        self.stride, self.padding = stride, padding
        self.output_padding, self.groups = output_padding, groups

    def __call__(self, x):
        y = mx.conv_transpose1d(x, self.w, stride=self.stride, padding=self.padding,
                                output_padding=self.output_padding, groups=self.groups)
        if self.b is not None:
            y = y + self.b
        return y


class Conv2dM:
    """Runtime Conv2d on channels-last (N,H,W,C). Weight already in MLX layout."""
    def __init__(self, w, b=None, stride=1, padding=0, dilation=1, groups=1):
        self.w, self.b = w, b
        self.stride, self.padding = stride, padding
        self.dilation, self.groups = dilation, groups

    def __call__(self, x):
        y = mx.conv2d(x, self.w, stride=self.stride, padding=self.padding,
                      dilation=self.dilation, groups=self.groups)
        if self.b is not None:
            y = y + self.b
        return y


def group_norm(x, num_groups, weight, bias, eps=1e-5):
    # x: (N, H, W, C) channels-last; torch GroupNorm semantics
    N, H, W, C = x.shape
    Cg = C // num_groups
    xg = x.reshape(N, H, W, num_groups, Cg)
    mean = mx.mean(xg, axis=(1, 2, 4), keepdims=True)
    var = mx.var(xg, axis=(1, 2, 4), keepdims=True)
    xg = (xg - mean) * mx.rsqrt(var + eps)
    x = xg.reshape(N, H, W, C)
    return x * weight + bias


def group_norm1d(x, num_groups, weight, bias, eps=1e-5):
    # x: (N, L, C) channels-last; torch GroupNorm semantics on 1D
    N, L, C = x.shape
    Cg = C // num_groups
    xg = x.reshape(N, L, num_groups, Cg)
    mean = mx.mean(xg, axis=(1, 3), keepdims=True)
    var = mx.var(xg, axis=(1, 3), keepdims=True)
    xg = (xg - mean) * mx.rsqrt(var + eps)
    x = xg.reshape(N, L, C)
    return x * weight + bias


def instance_norm1d(x, eps=1e-5):
    # x: (N, L, C); normalize each channel over time L per sample (no affine)
    mean = mx.mean(x, axis=1, keepdims=True)
    var = mx.var(x, axis=1, keepdims=True)
    return (x - mean) * mx.rsqrt(var + eps)


def reflect_pad1d(x, p):
    # x: (N, L, C); mirror without repeating the edge (torch 'reflect')
    if p == 0:
        return x
    left = x[:, 1:p + 1, :][:, ::-1, :]
    right = x[:, -p - 1:-1, :][:, ::-1, :]
    return mx.concatenate([left, x, right], axis=1)


def replicate_pad1d(x, left, right):
    # torch 'replicate' / edge padding
    parts = []
    if left > 0:
        parts.append(mx.broadcast_to(x[:, :1, :], (x.shape[0], left, x.shape[2])))
    parts.append(x)
    if right > 0:
        parts.append(mx.broadcast_to(x[:, -1:, :], (x.shape[0], right, x.shape[2])))
    return mx.concatenate(parts, axis=1) if len(parts) > 1 else x
