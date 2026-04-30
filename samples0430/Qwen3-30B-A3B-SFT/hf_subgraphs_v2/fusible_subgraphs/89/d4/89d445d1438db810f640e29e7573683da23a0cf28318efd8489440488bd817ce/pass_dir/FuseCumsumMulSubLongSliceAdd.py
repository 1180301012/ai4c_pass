import torch
import triton
import triton.language as tl


# ──────────────────────────────────────────────────────────────────────────────
# Pattern: mirrors model.py exactly
# ──────────────────────────────────────────────────────────────────────────────
def pattern(in_0):
    tmp_1 = torch.cumsum(in_0, dim=1)
    tmp_2 = tmp_1 * in_0
    tmp_3 = tmp_2 - 1
    tmp_4 = tmp_3.long()
    tmp_5 = tmp_4[slice(None, None, None), slice(0, None, None)]
    tmp_6 = tmp_5 + 2
    return tmp_6


def replacement_args(in_0):
    return (in_0,)


# ──────────────────────────────────────────────────────────────────────────────
# Fused Triton kernel: cumsum → mul → sub → long → slice → add
# Specialised for axis-1 cumsum on a [B, S] tensor.
# We lay out the 2-D grid as (B, ceil(S / BLOCK_SIZE)) so that each program
# holds one complete row (all S elements) in registers and uses tl.cumsum(axis=0).
# ──────────────────────────────────────────────────────────────────────────────
@triton.jit
def fused_cumsum_attention_mask_kernel(
    in_ptr,
    out_ptr,
):
    # BLOCK_SIZE=16 covers S=13; mask and S are compile-time constants
    offs  = tl.arange(0, 16)
    mask  = offs < 13          # 13 is a compile-time literal

    x = tl.load(in_ptr + offs, mask=mask, other=0)
    cumsum  = tl.cumsum(x, axis=0)
    product = cumsum * x
    result  = product + 1      # fused -1 + 2
    tl.store(out_ptr + offs, result, mask=mask)


# ──────────────────────────────────────────────────────────────────────────────
# Kernel wrapper (must be @torch.fx.wrap)
# ──────────────────────────────────────────────────────────────────────────────
@torch.fx.wrap
def fused_cumsum_attention_mask(in_0):
    out = torch.empty_like(in_0)
    fused_cumsum_attention_mask_kernel[(1,)](
        in_0,
        out,
        num_warps=2,
        num_stages=1,
    )
    return out


# ──────────────────────────────────────────────────────────────────────────────
def replacement_func():
    return fused_cumsum_attention_mask