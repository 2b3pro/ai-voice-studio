"""resemble-enhance-mlx: Apple MLX port of resemble-enhance (denoise + enhance).

    from resemble_enhance_mlx import load, enhance
    import soundfile as sf

    model = load("resemble-enhance-mlx.safetensors", lambd=0.5)
    wav, sr = sf.read("noisy.wav", dtype="float32")
    out, out_sr = enhance(model, wav, sr)
    sf.write("clean.wav", out, out_sr)
"""
from .infer import load, enhance

__all__ = ["load", "enhance"]
