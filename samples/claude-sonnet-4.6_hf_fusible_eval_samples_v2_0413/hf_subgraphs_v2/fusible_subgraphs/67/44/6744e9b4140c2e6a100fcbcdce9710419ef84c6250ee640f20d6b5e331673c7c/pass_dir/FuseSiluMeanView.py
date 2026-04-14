import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Pattern: mean((2,3)) on a 4-D tensor
# The silu is computed by the model BEFORE this op; we only replace the mean.
# The downstream view(1,1,-1) stays in the graph.
# ---------------------------------------------------------------------------
def pattern(in_1):
    tmp_1 = in_1.mean((2, 3))
    return tmp_1


def replacement_args(in_1):
    return (in_1,)


# ---------------------------------------------------------------------------
# Triton kernel: spatial mean over H*W per (B,C) channel, single-pass.
# Stores directly into the output buffer (same dtype as input) — no extra cast.
# Grid: (B*C,)  BLOCK_HW: next power-of-2 >= HW, chosen in wrapper.
# ---------------------------------------------------------------------------
@triton.jit
def _mean_hw_kernel(
    in_ptr,          # [B*C, HW]  input (any float dtype)
    out_ptr,         # [B*C]      output (same dtype as input)
    HW,
    BLOCK_HW: tl.constexpr,
):
    bc_idx = tl.program_id(0)
    base   = bc_idx * HW

    offs = tl.arange(0, BLOCK_HW)
    mask = offs < HW

    x     = tl.load(in_ptr + base + offs, mask=mask, other=0.0)
    x_f32 = x.to(tl.float32)

    # Mask invalid lanes before summing (valid=True → use x_f32, else 0)
    contrib = tl.where(mask, x_f32, tl.zeros([BLOCK_HW], dtype=tl.float32))
    total   = tl.sum(contrib)
    mean_v  = total / HW

    # Cast back to input dtype and store (eliminates a separate cast kernel)
    tl.store(out_ptr + bc_idx, mean_v.to(x.dtype))


# ---------------------------------------------------------------------------
# Wrapper: allocates [B, C] output directly in the input dtype, launches
# the mean kernel once, returns [B, C] tensor.
# BLOCK_HW is computed once per unique shape (no autotune overhead).
# ---------------------------------------------------------------------------
@torch.fx.wrap
def silu_mean_view_fused(in_1):
    B, C, H, W = in_1.shape
    HW = H * W
    BC = B * C

    # Pre-allocate output in the INPUT dtype → no extra cast kernel needed
    mean_out = torch.empty((B, C), dtype=in_1.dtype, device=in_1.device)

    # Choose BLOCK_HW as next power-of-2 (≥ HW, ≥ 64) for single-pass loading
    BLOCK_HW = max(triton.next_power_of_2(HW), 64)

    _mean_hw_kernel[(BC,)](
        in_1, mean_out,
        HW=HW,
        BLOCK_HW=BLOCK_HW,
    )

    return mean_out


# ---------------------------------------------------------------------------
# replacement_func: return the wrapper (not a call)
# ---------------------------------------------------------------------------
def replacement_func():
    return silu_mean_view_fused