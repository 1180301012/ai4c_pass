import torch
import triton
import triton.language as tl
from torch import device
from pass_dir.causal_mask_kernel import build_causal_mask_kernel


# ---------------------------------------------------------------------------
# Pattern: the 7-op constant causal-mask chain for N=21.
# All nodes produce only constant tensors (no external inputs).
# The output tmp_7 ([21,21]) is the only node with external consumers
# (getitem, expand, clone, …) — external consumers of the OUTPUT are OK.
# ---------------------------------------------------------------------------
def pattern():
    tmp_1 = torch.arange(0, 21, device=device(type='cuda', index=0))
    tmp_2 = torch.full((21, 21), fill_value=-3.4028234663852886e+38, dtype=torch.float32, device=device(type='cuda', index=0))
    tmp_3 = torch.triu(tmp_2, diagonal=1)
    tmp_4 = torch.arange(21, device=device(type='cuda', index=0))
    tmp_5 = tmp_1.reshape(-1, 1)
    tmp_6 = tmp_4 > tmp_5
    tmp_3 *= tmp_6
    tmp_7 = tmp_3
    return tmp_7


# ---------------------------------------------------------------------------
# Argument extraction (no pattern inputs)
# ---------------------------------------------------------------------------
def replacement_args():
    return ()


# ---------------------------------------------------------------------------
# Replacement: build the [21,21] causal mask with one Triton kernel
# ---------------------------------------------------------------------------
@torch.fx.wrap
def causal_mask_N21():
    N = 21
    out = torch.empty((N, N), dtype=torch.float32, device='cuda:0')
    build_causal_mask_kernel[(N,)](out, N=N, BLOCK_N=32)
    return out


def replacement_func():
    return causal_mask_N21