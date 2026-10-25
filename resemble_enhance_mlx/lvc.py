"""MLX port of the location-variable convolution (LVC) from UnivNet/LVCNet.

At all call sites dilation==1, so the torch unfold/transpose/unfold/einsum chain
reduces to: for output frame l and offset s, the input window is
x_padded[l*hop + s : l*hop + s + ks], convolved with the per-frame kernel.
"""
import mlx.core as mx


def location_variable_convolution(x, kernel, bias, hop_size):
    """
    x:      (B, C_in, in_length)  with in_length == kernel_length * hop_size
    kernel: (B, C_in, C_out, ks, kernel_length)
    bias:   (B, C_out, kernel_length)
    returns (B, C_out, in_length)
    """
    B, C_in, in_length = x.shape
    _, _, C_out, ks, KL = kernel.shape
    hop_size = int(hop_size)
    assert in_length == KL * hop_size, f"{in_length} != {KL}*{hop_size}"

    p = (ks - 1) // 2  # dilation == 1
    x = mx.pad(x, [(0, 0), (0, 0), (p, p)])  # (B, C_in, in_length + 2p)

    # window gather: idx[l, s, k] = l*hop + s + k
    l = (mx.arange(KL) * hop_size)[:, None, None]
    s = mx.arange(hop_size)[None, :, None]
    k = mx.arange(ks)[None, None, :]
    idx = l + s + k                       # (KL, hop, ks)
    xw = x[:, :, idx]                     # (B, C_in, KL, hop, ks)

    # o[b,o,l,s] = sum_{i,k} xw[b,i,l,s,k] * kernel[b,i,o,k,l]
    o = mx.einsum("bilsk,biokl->bols", xw, kernel)  # (B, C_out, KL, hop)
    o = o + bias[:, :, :, None]
    return o.reshape(B, C_out, KL * hop_size)
