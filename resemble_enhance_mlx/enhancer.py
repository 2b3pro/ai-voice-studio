"""End-to-end MLX resemble-enhance: mel -> normalizer -> latent CFM -> vocoder.

Reproduces resemble_enhance Enhancer.forward in eval/cfm mode. Optional denoiser
path (lambd > 0). tau_noise / voc_noise inputs exist only for parity testing;
left None they draw fresh randn at runtime (as the torch reference does).
"""
import mlx.core as mx
from .mel import MelMLX
from .lcfm import LCFMM, NormalizerM
from .vocoder import UnivNetM, cf2cl, cl2cf
from .denoiser import DenoiserM


def _normalize_wav(x):  # x: (..., T)
    return x / (mx.max(mx.abs(x), axis=-1, keepdims=True) + 1e-7)


class EnhancerM:
    def __init__(self, t_enh, fb_path, window_path, lambd=0.5):
        self.mel = MelMLX(fb_path, window_path)
        self.normalizer = NormalizerM(t_enh.normalizer)
        self.lcfm = LCFMM(t_enh.lcfm)
        self.vocoder = UnivNetM(t_enh.vocoder)
        self.denoiser = DenoiserM(t_enh.denoiser)
        self.lambd = lambd

    def to_mel(self, wav):  # wav: (T,) -> (1, NUM_MELS, t) channels-first, last frame dropped
        m = self.mel(wav)[:, :-1]
        return m[None]

    def __call__(self, wav, tau_noise=None, voc_noise=None):  # wav: (T,)
        x = _normalize_wav(wav)
        x_mel_original = self.normalizer(self.to_mel(x))
        if self.lambd == 0:
            x_mel_denoised = x_mel_original
        else:
            dn = self.denoiser(x[None])[0]
            x_mel_d = self.normalizer(self.to_mel(dn))
            x_mel_denoised = self.lambd * x_mel_d + (1 - self.lambd) * x_mel_original

        xcl = cf2cl(x_mel_denoised)
        p0cl = cf2cl(x_mel_original)
        dec = self.lcfm(xcl, p0cl, tau_noise=tau_noise)   # (1, t, 160)
        feat = cl2cf(dec)                                 # (1, 160, t)
        o = self.vocoder(feat, noise=voc_noise)           # (1, T2)
        return o
