"""MLX port of UnivNet/LVCNet vocoder."""
import numpy as np
import mlx.core as mx
from .convert import conv1d_from_torch, convt1d_from_torch
from .amp import AMPBlockM
from .lvc import location_variable_convolution as lvc


def lrelu(x, slope=0.2):
    return mx.where(x >= 0, x, slope * x)


def cl2cf(x):  # (N,L,C)->(N,C,L)
    return mx.transpose(x, (0, 2, 1))


def cf2cl(x):
    return mx.transpose(x, (0, 2, 1))


class KernelPredictorM:
    def __init__(self, t_kp):
        self.conv_in_channels = t_kp.conv_in_channels
        self.conv_out_channels = t_kp.conv_out_channels
        self.conv_kernel_size = t_kp.conv_kernel_size
        self.conv_layers = t_kp.conv_layers
        self.slope = t_kp.input_conv[1].negative_slope
        self.input_conv = conv1d_from_torch(t_kp.input_conv[0], padding=2)
        self.residual = []
        for rc in t_kp.residual_convs:  # Seq(Dropout, Conv, LReLU, Conv, LReLU)
            self.residual.append((conv1d_from_torch(rc[1], padding=1),
                                  conv1d_from_torch(rc[3], padding=1)))
        self.kernel_conv = conv1d_from_torch(t_kp.kernel_conv, padding=1)
        self.bias_conv = conv1d_from_torch(t_kp.bias_conv, padding=1)

    def __call__(self, c):  # c: (N,L,cond) channels-last
        B, L, _ = c.shape
        c = lrelu(self.input_conv(c), self.slope)
        for c1, c2 in self.residual:
            h = lrelu(c1(c), self.slope)
            h = lrelu(c2(h), self.slope)
            c = c + h
        k = cl2cf(self.kernel_conv(c))   # (N, Ck, L)
        b = cl2cf(self.bias_conv(c))     # (N, Cb, L)
        kernels = k.reshape(B, self.conv_layers, self.conv_in_channels,
                            self.conv_out_channels, self.conv_kernel_size, L)
        bias = b.reshape(B, self.conv_layers, self.conv_out_channels, L)
        return kernels, bias


class LVCBlockM:
    def __init__(self, t_block):
        self.cond_hop_length = t_block.cond_hop_length
        self.in_channels = t_block.amp_block[0][0].in_channels  # conv channels
        cp = t_block.convt_pre
        self.cp_slope = cp[0].negative_slope
        # stride==1 -> Conv1d(1); else ConvTranspose1d
        import torch.nn as nn
        if isinstance(cp[1], nn.ConvTranspose1d) or hasattr(cp[1], "output_padding"):
            self.convt_pre = convt1d_from_torch(
                cp[1], stride=cp[1].stride[0], padding=cp[1].padding[0],
                output_padding=cp[1].output_padding[0])
            self.is_transpose = True
        else:
            self.convt_pre = conv1d_from_torch(cp[1])
            self.is_transpose = False
        self.amp_block = AMPBlockM(t_block.amp_block)
        self.kernel_predictor = KernelPredictorM(t_block.kernel_predictor)
        self.conv_blocks = []
        for cb in t_block.conv_blocks:  # Seq(LReLU, Conv(dilation,same), LReLU)
            d = cb[1].dilation[0]
            self.conv_blocks.append((conv1d_from_torch(cb[1], padding=d, dilation=d), cb[0].negative_slope))

    def __call__(self, x, c):  # x,c channels-last (N,L,C)
        x = self.convt_pre(lrelu(x, self.cp_slope))
        x = self.amp_block(x)
        kernels, bias = self.kernel_predictor(c)
        ic = self.in_channels
        for i, (conv, slope) in enumerate(self.conv_blocks):
            out = lrelu(conv(lrelu(x, slope)), slope)   # (N,L,C)
            out_cf = cl2cf(out)                          # (N,C,L)
            k = kernels[:, i]
            b = bias[:, i]
            o = lvc(out_cf, k, b, hop_size=self.cond_hop_length)  # (N, 2C, L)
            o = cf2cl(o)                                  # (N,L,2C)
            gate = mx.sigmoid(o[:, :, :ic]) * mx.tanh(o[:, :, ic:])
            x = x + gate
        return x


class UnivNetM:
    def __init__(self, t_voc):
        self.d_noise = t_voc.d_noise
        self.scale_factor = t_voc.scale_factor
        self.conv_pre = conv1d_from_torch(t_voc.conv_pre, padding=3, pad_mode="reflect")
        self.blocks = [LVCBlockM(b) for b in t_voc.blocks]
        self.post_slope = t_voc.conv_post[0].negative_slope
        self.conv_post = conv1d_from_torch(t_voc.conv_post[1], padding=3, pad_mode="reflect")

    def __call__(self, x, npad=10, noise=None):
        # x: (N, C, T) channels-first features (matches torch API)
        x = mx.pad(x, [(0, 0), (0, 0), (0, npad)])
        N, _, T = x.shape
        if noise is None:
            noise = mx.random.normal((N, self.d_noise, T))
        z = cf2cl(noise)
        x = cf2cl(x)
        z = self.conv_pre(z)
        for block in self.blocks:
            z = block(z, x)
        z = mx.tanh(self.conv_post(lrelu(z, self.post_slope)))  # (N,L,1)
        z = cl2cf(z)[:, 0, :]                                    # (N, L)
        return z[:, : -self.scale_factor * npad]
