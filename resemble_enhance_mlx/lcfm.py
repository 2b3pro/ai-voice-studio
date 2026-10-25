"""MLX port of resemble-enhance's latent CFM enhancer (IRMAE + WN + ODE solver).

Activations are carried channels-last (B, T, C). The CFM ODE is solved with the
midpoint method over an exponential-decay time schedule (computed offline in
numpy, no torch at runtime).
"""
import math
import numpy as np
import mlx.core as mx
import scipy.optimize
from .nn import group_norm1d, instance_norm1d
from .convert import conv1d_from_torch, _np
from .denoiser import gelu


def _gn(t_gn):
    return (_np(t_gn.weight), _np(t_gn.bias))


# ---------------- IRMAE autoencoder ----------------

class _ResBlock:
    def __init__(self, t_rb, channels, dilations=(1, 2, 4, 8)):
        self.ng = 32
        self.norms = []
        self.convs = []
        for j, d in enumerate(dilations):
            self.norms.append(_gn(t_rb[3 * j]))            # GroupNorm
            self.convs.append(conv1d_from_torch(t_rb[3 * j + 2], padding=d, dilation=d))

    def __call__(self, x):
        h = x
        for (gw, gb), conv in zip(self.norms, self.convs):
            h = group_norm1d(h, self.ng, gw, gb)
            h = conv(gelu(h))
        return x + h


class IRMAEM:
    def __init__(self, t_ae):
        enc = t_ae.encoder
        self.enc_in = conv1d_from_torch(enc[0], padding=1)
        self.enc_res = [_ResBlock(enc[1 + i], enc[1 + i][2].out_channels) for i in range(4)]
        self.enc_irm = [conv1d_from_torch(enc[5 + i], padding=0) for i in range(4)]
        dec = t_ae.decoder
        self.dec_in = conv1d_from_torch(dec[0], padding=1)
        self.dec_res = [_ResBlock(dec[1 + i], dec[1 + i][2].out_channels) for i in range(4)]
        self.dec_out = conv1d_from_torch(dec[5], padding=0)

    def encode(self, x):  # x: (B, T, C_in)
        z = self.enc_in(x)
        for rb in self.enc_res:
            z = rb(z)
        for conv in self.enc_irm:
            z = conv(z)
        return mx.tanh(z)

    def decode(self, z):  # z: (B, T, latent)
        h = self.dec_in(z)
        for rb in self.dec_res:
            h = rb(h)
        return self.dec_out(h)


# ---------------- WN velocity net ----------------

class _WNLayer:
    def __init__(self, t_layer, hidden):
        self.hidden = hidden
        self.gconv = conv1d_from_torch(t_layer.gconv, padding=0)
        self.lconv = conv1d_from_torch(t_layer.lconv, padding=0)
        d = t_layer.dconv.dilation[0]
        self.dconv = conv1d_from_torch(t_layer.dconv, padding=d, dilation=d)
        self.out = conv1d_from_torch(t_layer.out, padding=0)

    def __call__(self, z, l, g):  # z,l: (B,T,H)/(B,T,local); g: (B,1,gdim)
        identity = z
        z = z + self.gconv(g)
        z = self.dconv(z)
        z = z + self.lconv(l)
        a = mx.tanh(z[..., :self.hidden])
        b = mx.sigmoid(z[..., self.hidden:])
        z = a * b
        h = self.out(z)
        zz = h[..., :self.hidden]
        s = h[..., self.hidden:]
        o = (zz + identity) / math.sqrt(2)
        return o, s


class _WN:
    def __init__(self, t_wn):
        self.hidden = t_wn.hidden_dim
        self.start = conv1d_from_torch(t_wn.start, padding=0)
        self.layers = [_WNLayer(layer, self.hidden) for layer in t_wn.layers]
        self.end = conv1d_from_torch(t_wn.end, padding=0)

    def __call__(self, z, l, g):  # z: (B,T,in); l: (B,T,local); g: (B,1,gdim)
        z = self.start(z)
        l = instance_norm1d(l)
        s_sum = None
        for layer in self.layers:
            z, s = layer(z, l, g)
            s_sum = s if s_sum is None else s_sum + s
        s_sum = s_sum / math.sqrt(len(self.layers))
        return self.end(s_sum)


def _sinusoidal_emb(t, d_embed):  # t: (B,) -> (B, d_embed)
    p = mx.linspace(0, 4, d_embed // 2)
    tt = t[:, None]
    sin = mx.sin(tt * 10 ** p)
    cos = mx.cos(tt * 10 ** p)
    return mx.concatenate([sin, cos], axis=-1)


class CFMM:
    def __init__(self, t_cfm):
        self.net = _WN(t_cfm.net)
        self.time_emb_dim = t_cfm.emb.d_embed
        self.nfe = t_cfm.solver.nfe
        self.method = t_cfm.solver.method
        self.divisor = t_cfm.solver._time_mapping.keywords["n"]

    @property
    def n_steps(self):
        if self.method == "euler":
            return self.nfe
        if self.method == "midpoint":
            return self.nfe // 2
        if self.method == "rk4":
            return self.nfe // 4
        raise ValueError(self.method)

    def _time_mapping(self, t):
        n = self.divisor
        a = float(scipy.optimize.fsolve(lambda a: ((a ** (1 / n) - 1) / (a - 1)) - 0.5, x0=0)[0])
        return (a ** t - 1) / (a - 1)

    def _to_v(self, pst, x, t):  # pst,x: (B,T,C); t: float
        B = pst.shape[0]
        tt = mx.clip(mx.full((B,), t), 0.0, 1.0)
        g = _sinusoidal_emb(tt, self.time_emb_dim)[:, None, :]   # (B,1,gdim)
        return self.net(pst, x, g)

    def sample(self, x, psi0):  # midpoint ODE from t0=0 to 1
        ts = self._time_mapping(np.linspace(0.0, 1.0, self.n_steps + 1))
        pst = psi0
        for i in range(self.n_steps):
            t = float(ts[i])
            dt = float(ts[i + 1] - ts[i])
            if self.method == "euler":
                pst = pst + dt * self._to_v(pst, x, t)
            else:  # midpoint
                k1 = self._to_v(pst, x, t)
                pst = pst + dt * self._to_v(pst + dt * k1 / 2, x, t + dt / 2)
        return pst


# ---------------- LCFM wrapper ----------------

class LCFMM:
    def __init__(self, t_lcfm):
        self.ae = IRMAEM(t_lcfm.ae)
        self.cfm = CFMM(t_lcfm.cfm)
        self.z_scale = t_lcfm.z_scale
        self.eval_tau = t_lcfm._eval_tau

    def __call__(self, x, psi0, tau_noise=None):
        """x, psi0: (B, T, C) channels-last mels. tau_noise: injected randn for parity."""
        p0 = self.ae.encode(psi0) * self.z_scale
        tau = self.eval_tau
        if tau_noise is None:
            tau_noise = mx.random.normal(p0.shape)
        p0 = tau * tau_noise + (1 - tau) * p0
        z = self.cfm.sample(x, p0) / self.z_scale
        return self.ae.decode(z)


class NormalizerM:
    def __init__(self, t_norm):
        self.mean = float(t_norm.running_mean.item())
        self.std = float(t_norm.running_std.item())

    def __call__(self, x):
        return (x - self.mean) / self.std
