"""Generic (de)serialization of a built MLX module graph to a single safetensors.

Walks the object graph of an already-constructed module (e.g. EnhancerM), saving
every mx.array / np.ndarray leaf into one flat safetensors file and the structural
tree (classes, scalars, containers) as JSON metadata. Reconstruction rebuilds the
same objects via __new__ + attribute restore — no torch needed at load time.
"""
import json
import importlib
import numpy as np
import mlx.core as mx


def _flatten(obj, prefix, arrays):
    if isinstance(obj, mx.array):
        arrays[prefix] = obj
        return {"l": "mx"}
    if isinstance(obj, np.ndarray):
        arrays[prefix] = mx.array(obj)
        return {"l": "np", "dt": str(obj.dtype)}
    if isinstance(obj, np.integer):
        return {"v": int(obj)}
    if isinstance(obj, np.floating):
        return {"v": float(obj)}
    if isinstance(obj, bool) or obj is None or isinstance(obj, (int, float, str)):
        return {"v": obj}
    if isinstance(obj, (list, tuple)):
        return {"s": type(obj).__name__,
                "i": [_flatten(v, f"{prefix}.{i}", arrays) for i, v in enumerate(obj)]}
    if hasattr(obj, "__dict__"):
        cls = type(obj)
        return {"o": f"{cls.__module__}:{cls.__qualname__}",
                "a": {k: _flatten(v, f"{prefix}.{k}", arrays) for k, v in vars(obj).items()}}
    raise TypeError(f"cannot serialize {prefix}: {type(obj)}")


def _build(meta, prefix, arrays):
    if "l" in meta:
        a = arrays[prefix]
        return np.array(a).astype(meta["dt"]) if meta["l"] == "np" else a
    if "v" in meta:
        return meta["v"]
    if "s" in meta:
        items = [_build(m, f"{prefix}.{i}", arrays) for i, m in enumerate(meta["i"])]
        return tuple(items) if meta["s"] == "tuple" else items
    if "o" in meta:
        modname, qual = meta["o"].split(":")
        # remap the stored package root to whatever package this loader lives in
        root = __package__.split(".")[0]
        modname = root + modname[modname.index("."):] if "." in modname else modname
        cls = getattr(importlib.import_module(modname), qual)
        inst = cls.__new__(cls)
        for k, m in meta["a"].items():
            setattr(inst, k, _build(m, f"{prefix}.{k}", arrays))
        return inst
    raise ValueError(f"bad meta at {prefix}: {meta}")


# Computed constants (not from the fp16 checkpoint) must stay fp32 for accuracy.
def _keep_fp32(key):
    return key.endswith((".mel.fb", ".mel.window", "._filt_np"))


def save(obj, path, fp16=True):
    arrays = {}
    tree = _flatten(obj, "root", arrays)
    if fp16:
        arrays = {k: (v.astype(mx.float16) if v.dtype == mx.float32 and not _keep_fp32(k) else v)
                  for k, v in arrays.items()}
    mx.save_safetensors(path, arrays, metadata={"tree": json.dumps(tree)})


def load(path):
    arrays, md = mx.load(path, return_metadata=True)
    arrays = {k: (v.astype(mx.float32) if v.dtype == mx.float16 else v) for k, v in arrays.items()}
    tree = json.loads(md["tree"])
    return _build(tree, "root", arrays)
