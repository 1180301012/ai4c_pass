import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Pattern: sum along dim=2 (keepdim=True) followed by element-wise division
# Matches: tmp_0 = in_1.sum(dim=2, keepdim=True)
#          tmp_1 = in_1 / tmp_0
# ---------------------------------------------------------------------------
def pattern(x):
    s = x.sum(dim=2, keepdim=True)
    y = x / s
    return y


def replacement_args(x):
    return (x,)


# ---------------------------------------------------------------------------
# Fused Triton kernel — 2-D per-(b,c)-slice design
#
# For [B, C, D, W]: Grid=(B*C,), one CTA per (b,c) slice.
# Each CTA:
#   - loads [BLOCK_D, BLOCK_W] contiguous tile  (coalesced: stride_w=1)
#   - reduces along D (axis 0) → column sums [W]
#   - divides in fp32 for precision, stores as original dtype
# ---------------------------------------------------------------------------
@triton.jit
def fused_sum_div_kernel(
    x_ptr,
    out_ptr,
    B,
    C,
    D,
    W,
    stride_b,
    stride_c,
    stride_d,
    stride_w,
    BLOCK_D: tl.constexpr,
    BLOCK_W: tl.constexpr,
):
    pid  = tl.program_id(0)        # (b,c) index  [0, B*C)
    b    = pid // C
    c    = pid % C
    base = b * stride_b + c * stride_c

    d_idx = tl.arange(0, BLOCK_D)[:, None]    # [D, 1]
    w_idx = tl.arange(0, BLOCK_W)[None, :]    # [1, W]

    mask    = (d_idx < D) & (w_idx < W)
    offsets = base + d_idx * stride_d + w_idx * stride_w    # [D, W]

    x_tile  = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    x_f32   = x_tile.to(tl.float32)
    col_sum = tl.sum(x_f32, axis=0)           # [W]
    out_f32 = x_f32 / col_sum[None, :]
    out_tile = out_f32.to(x_tile.dtype)
    tl.store(out_ptr + offsets, out_tile, mask=mask)


# ---------------------------------------------------------------------------
# Dispatch threshold: below this element count, Triton's Python-side kernel
# launch overhead exceeds the savings from fusing sum+div into one kernel.
# PyTorch's pre-compiled CUDA kernels have lower effective overhead for tiny
# tensors.  Above the threshold, Triton wins because it avoids writing the
# intermediate tmp_0 to global memory and reduces to one kernel launch.
# ---------------------------------------------------------------------------
_TRITON_MIN_NUMEL = 1024


@torch.fx.wrap
def fused_sum_div(x):
    if x.numel() >= _TRITON_MIN_NUMEL:
        # --- Triton path (large tensors: one fused kernel) --------------------
        B, C, D, W = x.shape
        out     = torch.empty_like(x)
        BLOCK_D = triton.next_power_of_2(D)
        BLOCK_W = triton.next_power_of_2(W)
        fused_sum_div_kernel[(B * C,)](
            x, out,
            B, C, D, W,
            x.stride(0), x.stride(1), x.stride(2), x.stride(3),
            BLOCK_D=BLOCK_D,
            BLOCK_W=BLOCK_W,
            num_warps=1,
            num_stages=1,
        )
        return out
    # --- PyTorch eager path (small tensors: lower dispatch overhead) ----------
    return x / x.sum(dim=2, keepdim=True)


# ---------------------------------------------------------------------------
# replacement_func: zero-argument factory — returns the wrapper function
# ---------------------------------------------------------------------------
def replacement_func():
    return fused_sum_div