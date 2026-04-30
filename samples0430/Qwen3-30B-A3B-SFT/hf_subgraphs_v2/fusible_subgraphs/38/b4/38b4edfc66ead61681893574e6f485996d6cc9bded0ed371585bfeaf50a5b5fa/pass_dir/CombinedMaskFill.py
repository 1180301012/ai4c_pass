import torch
import triton
import triton.language as tl
from torch import device


@triton.jit
def _combined_mask_fill_kernel(
    x_ptr,   # int64 [N] – input mask values (0 or 1)
    z_ptr,   # float32 [N*N] – causal mask values
    out_ptr, # float32 [N*N]
    N,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fused: out = (z == 0.0) & (x == 0)  →  0.0 if both unmasked, else -3.4e38
    z is the causal mask (float32, -3.4e38 or 0.0), x is the input mask (int64, 0 or 1).
    """
    NEG_INF: tl.constexpr = -3.4028234663852886e+38

    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    total = N * N
    valid = offsets < total

    col = offsets % N

    z_val = tl.load(z_ptr + offsets, mask=valid, other=NEG_INF)
    x_val = tl.load(x_ptr + col, mask=valid, other=1)

    keep = (z_val == 0.0) & (x_val == 0)
    result = tl.where(keep, 0.0, NEG_INF)

    tl.store(out_ptr + offsets, result, mask=valid)


@torch.fx.wrap
def combined_mask_fill_dispatch(x, z):
    """
    x: float32 [1,1,N,N] – pre-computed (1.0 - in_0_float) mask
    z: float32 [1,1,N,N] – causal mask
    """
    N = z.shape[2]
    out = torch.empty_like(z)
    BLOCK_SIZE = 128
    total = N * N
    num_blocks = (total + BLOCK_SIZE - 1) // BLOCK_SIZE
    _combined_mask_fill_kernel[(num_blocks,)](x, z, out, N, BLOCK_SIZE=BLOCK_SIZE)
    return out


# ─── Pattern: fuse the full input-mask post-processing chain ─────────────────
# x = tmp_14 (float32), z = tmp_9 (causal mask)

def pattern(x, z):
    tmp_15 = x.to(torch.bool)
    tmp_16 = x.masked_fill(tmp_15, -3.4028234663852886e+38)
    tmp_17 = tmp_16.to(device(type='cuda', index=0))
    tmp_18 = tmp_17.bool()
    tmp_19 = z.masked_fill(tmp_18, -3.4028234663852886e+38)
    return tmp_19


def replacement_args(x, z):
    return (x, z)


def replacement_func():
    return combined_mask_fill_dispatch