"""
Shared Triton kernel: fused SiLU + global average pool + flatten.
Input shape: [N, C, H, W]  →  Output shape: [N, C]
dropout(training=False) is a no-op, so that output IS the result.
"""

import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_HW': 64},  num_warps=2, num_stages=2),
        triton.Config({'BLOCK_HW': 64},  num_warps=4, num_stages=2),
        triton.Config({'BLOCK_HW': 128}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_HW': 128}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_HW': 256}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_HW': 256}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_HW': 512}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_HW': 512}, num_warps=16, num_stages=2),
    ],
    key=['HW', 'dtype_id'],
)
@triton.jit
def _silu_global_avg_kernel(
    x_ptr, out_ptr,
    HW, HW_log2,
    dtype_id,          # 0=float16, 1=bfloat16, 2=float32
    BLOCK_HW: tl.constexpr,
):
    # One program per (n, c) pair
    pid = tl.program_id(0)

    base     = pid * HW
    hw_off   = tl.arange(0, BLOCK_HW)
    mask     = hw_off < HW

    # Load spatial window (pad out-of-bounds with 0 → silu(0)=0, correct)
    x_orig = tl.load(x_ptr + base + hw_off, mask=mask, other=0.0)
    x      = x_orig.to(tl.float32)

    # SiLU: x * sigmoid(x)
    silu_x = x * tl.sigmoid(x)

    # Reduce: mean over HW spatial positions
    total  = tl.sum(silu_x, axis=0) / HW

    # Store with original dtype (cast from float32 inside JIT)
    if dtype_id == 0:
        tl.store(out_ptr + pid, total.to(tl.float16))
    elif dtype_id == 1:
        tl.store(out_ptr + pid, total.to(tl.bfloat16))
    else:
        tl.store(out_ptr + pid, total)


@torch.fx.wrap
def dispatch_silu_avgpool(x, route):
    """
    Common dispatch wrapper used by all dropout-ratio passes.
    `route` is appended by replacement_args() in each pass file and
    is not used for dispatch — the output is always the same computation.
    """
    N, C, H, W = x.shape
    HW = H * W
    NC = N * C

    out = torch.empty((N, C), dtype=x.dtype, device=x.device)

    dtype_id = 0 if x.dtype == torch.float16 else (1 if x.dtype == torch.bfloat16 else 2)

    _silu_global_avg_kernel[(NC,)](
        x, out,
        HW, HW,    # note: HW is used twice but that's fine
        dtype_id,
    )

    return out