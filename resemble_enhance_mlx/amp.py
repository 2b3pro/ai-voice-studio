"""MLX AMPBlock (BigVGAN anti-aliased multi-periodicity block)."""
import mlx.core as mx
from .alias_free import SnakeBeta, UpActDown
from .convert import conv1d_from_torch, _np


class _Layer:
    def __init__(self, t_layer, dilation):
        # t_layer = Sequential(Conv1d(same,dilation), UpActDown(SnakeBeta), Conv1d(same))
        self.conv1 = conv1d_from_torch(t_layer[0], padding=dilation, dilation=dilation)
        uad = t_layer[1]
        snake = SnakeBeta(_np(uad.act.log_alpha), _np(uad.act.log_beta))
        self.uad = UpActDown(snake,
                             up_ratio=uad.upsample.ratio, down_ratio=uad.downsample.lowpass.stride,
                             up_kernel_size=uad.upsample.kernel_size,
                             down_kernel_size=uad.downsample.lowpass.kernel_size)
        self.conv2 = conv1d_from_torch(t_layer[2], padding=1)

    def __call__(self, x):
        return self.conv2(self.uad(self.conv1(x)))


class AMPBlockM:
    def __init__(self, t_amp, dilations=(1, 3, 5)):
        self.layers = [_Layer(t_amp[i], d) for i, d in enumerate(dilations)]

    def __call__(self, x):  # x: (N, L, C)
        out = x
        for layer in self.layers:
            out = layer(out)
        return x + out
