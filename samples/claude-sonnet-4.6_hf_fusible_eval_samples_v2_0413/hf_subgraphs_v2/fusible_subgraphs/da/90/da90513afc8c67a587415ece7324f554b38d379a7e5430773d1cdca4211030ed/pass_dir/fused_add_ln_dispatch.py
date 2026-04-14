"""
Shared fused add + layer_norm Triton kernel and dispatch wrapper.
All pass files import `fused_add_layernorm_dispatch` from here so they
share the SAME function object — satisfying the replacement_func_limit.
"""

import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Single Triton kernel: BLOCK_SIZE is the only constexpr.
# Caller passes the correct BLOCK_SIZE (>= N) for each route.
# One-shot variance: var = E[z²] − E[z]²  → no tl.where on diff needed.
# ---------------------------------------------------------------------------

@triton.jit
def _fused_add_ln_kernel(
    X_ptr, Y_ptr, W_ptr, B_ptr, Out_ptr,
    M, eps,
    N: tl.constexpr,          # normalized dim — constexpr lets compiler fold static mask
    BLOCK_SIZE: tl.constexpr,  # next power-of-2 >= N
):
    """
    One CUDA block per input row.
    out[row] = layer_norm(X[row] + Y[row], W, B, eps)

    - Add is in native dtype (matches PyTorch's in_2 + in_3 exactly).
    - Statistics accumulate in float32 for precision.
    - N and BLOCK_SIZE are constexpr so the compiler can fold the static mask
      (positions cols >= N are always 0.0) and potentially optimise reductions.
    """
    row_id    = tl.program_id(0)
    row_start = row_id * N

    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < N                # static mask since N is constexpr

    # Load inputs (padded positions → 0.0)
    x = tl.load(X_ptr + row_start + cols, mask=mask, other=0.0)
    y = tl.load(Y_ptr + row_start + cols, mask=mask, other=0.0)
    # Add in native dtype first (matches PyTorch's `in_2 + in_3` op exactly),
    # then upcast to float32 for numerically-stable LN statistics.
    z_raw = x + y
    z     = z_raw.to(tl.float32)

    # One-shot mean + variance (padded positions have z=0.0, don't contribute)
    sum_z    = tl.sum(z, axis=0)
    mean     = sum_z / N
    sum_z_sq = tl.sum(z * z, axis=0)
    var      = tl.maximum(sum_z_sq / N - mean * mean, 0.0)

    # Normalise and affine-transform
    rstd   = 1.0 / tl.sqrt(var + eps)
    z_norm = (z - mean) * rstd
    w      = tl.load(W_ptr + cols, mask=mask, other=1.0).to(tl.float32)
    b      = tl.load(B_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    out    = z_norm * w + b

    tl.store(Out_ptr + row_start + cols, out.to(x.dtype), mask=mask)


# ---------------------------------------------------------------------------
# Dispatch wrapper — returned by replacement_func() in ALL pass files
# ---------------------------------------------------------------------------

@torch.fx.wrap
def fused_add_layernorm_dispatch(in_0, in_1, in_2, in_3, route):
    """
    in_0 = bias  [N], in_1 = weight [N]
    in_2/in_3 = hidden states [..., N]
    route: "768" | "768_rev" | "1024" | "16"

    Fixed parameters chosen empirically for NVIDIA A30:
      N=768  → BLOCK_SIZE=1024 (25% padding), num_warps=4
      N=1024 → BLOCK_SIZE=1024 (exact fit),   num_warps=4
      N=16   → BLOCK_SIZE=32  (1 full warp),  num_warps=1
    """
    if route == "768" or route == "768_rev":
        N, BS, nw = 768, 1024, 4
        x, y = in_2, in_3
    elif route == "1024":
        N, BS, nw = 1024, 1024, 4
        x, y = in_2, in_3
    elif route == "16":
        N, BS, nw = 16, 32, 1
        x, y = in_2, in_3
    else:
        N, BS, nw = 768, 1024, 4
        x, y = in_2, in_3

    M   = x.numel() // N
    out = torch.empty_like(x)
    _fused_add_ln_kernel[(M,)](
        x, y, in_1, in_0, out,
        M=M, eps=1e-5,
        N=N, BLOCK_SIZE=BS,
        num_warps=nw,
    )
    return out