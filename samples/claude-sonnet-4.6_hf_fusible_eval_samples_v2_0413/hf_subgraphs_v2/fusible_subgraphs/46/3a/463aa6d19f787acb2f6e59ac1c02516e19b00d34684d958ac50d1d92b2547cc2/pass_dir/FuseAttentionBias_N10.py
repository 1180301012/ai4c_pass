import torch
import triton
import triton.language as tl
from torch import device
from pass_dir.attention_bias_kernel import causal_attention_bias_kernel


# ---------------------------------------------------------------------------
# Pattern: exact mirror of the N=10 model.py forward
# ---------------------------------------------------------------------------
def pattern(in_0):
    tmp_1 = torch.arange(0, 10, device=device(type='cuda', index=0))
    tmp_2 = torch.full((10, 10), fill_value=-3.4028234663852886e+38, dtype=torch.float32, device=device(type='cuda', index=0))
    tmp_3 = torch.triu(tmp_2, diagonal=1)
    tmp_4 = torch.arange(10, device=device(type='cuda', index=0))
    tmp_5 = tmp_1.reshape(-1, 1)
    tmp_6 = tmp_4 > tmp_5
    tmp_3 *= tmp_6
    tmp_7 = tmp_3
    tmp_8 = tmp_7[(None, None, slice(None, None, None), slice(None, None, None))]
    tmp_9 = tmp_8.expand(1, 1, -1, -1)
    tmp_10 = tmp_9.clone()
    tmp_11 = tmp_10[(slice(None, None, None), slice(None, None, None), slice(None, None, None), slice(None, 10, None))]
    tmp_12 = in_0[(slice(None, None, None), None, None, slice(None, None, None))]
    tmp_13 = tmp_12.to(device(type='cuda', index=0))
    tmp_14 = tmp_11 + tmp_13
    tmp_15 = tmp_14.__eq__(0)
    tmp_16 = tmp_10[(slice(None, None, None), slice(None, None, None), slice(None, None, None), slice(None, 10, None))]
    tmp_17 = tmp_16.masked_fill(tmp_15, -3.4028234663852886e+38)
    # The _decomposed graph converts the in-place setitem to a functional clone:
    # tmp_10[:,:,:,:10] = tmp_17  →  tmp_10 = tmp_17.clone()
    tmp_10 = tmp_17.clone()
    tmp_19 = tmp_10.__eq__(-3.4028234663852886e+38)
    tmp_20 = torch.all(tmp_19, dim=-1, keepdim=True)
    tmp_21 = ~tmp_20
    tmp_22 = tmp_10.mul(tmp_21)
    return tmp_22


# ---------------------------------------------------------------------------
# Argument extraction
# ---------------------------------------------------------------------------
def replacement_args(in_0):
    return (in_0,)


# ---------------------------------------------------------------------------
# Optimised kernel wrapper
# ---------------------------------------------------------------------------
@torch.fx.wrap
def causal_attention_bias_N10(in_0):
    N = 10
    BLOCK_N = 16            # next power-of-2 >= N
    out = torch.empty((1, 1, N, N), dtype=torch.float32, device=in_0.device)
    causal_attention_bias_kernel[(N,)](
        in_0,
        out,
        N=N,
        BLOCK_N=BLOCK_N,
    )
    return out


def replacement_func():
    return causal_attention_bias_N10