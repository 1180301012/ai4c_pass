import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Pattern: elementwise-add followed by layer_norm (normalized_shape=(384,))
# The layer_norm result (tmp_6) is observable (returned by the model).
# ---------------------------------------------------------------------------

def pattern(in_5, in_6, in_1, in_2):
    tmp_5 = in_6 + in_5
    tmp_6 = torch.nn.functional.layer_norm(tmp_5, (384,), in_2, in_1, 1e-12)
    return tmp_6


def replacement_args(in_5, in_6, in_1, in_2):
    return (in_5, in_6, in_1, in_2)


# ---------------------------------------------------------------------------
# Combined-reduction combine function
#   Reduces (sum, sum_sq) pairs simultaneously → ONE inter-warp barrier
#   instead of two separate tl.sum calls.
# ---------------------------------------------------------------------------

@triton.jit
def _sum_sq_combine(s1_a, s2_a, s1_b, s2_b):
    return s1_a + s1_b, s2_a + s2_b


# ---------------------------------------------------------------------------
# Triton kernel: fused add + layer-norm
#
#   BLOCK_SIZE=512 (next power-of-2 ≥ D=384), num_warps=4.
#   Masked loads give 0.0 at padded positions → no tl.where needed.
#   Single tl.reduce call computes Σx and Σx² together (one sync barrier).
# ---------------------------------------------------------------------------

@triton.jit
def fused_add_layernorm_kernel(
    x_ptr, z_ptr,   # addends [N, D] row-major
    w_ptr, b_ptr,   # γ [D], β [D]
    out_ptr,        # output [N, D]
    D,              # row length  (384)
    eps,            # 1e-12
    BLOCK_SIZE: tl.constexpr,   # 512
):
    row     = tl.program_id(0)
    row_off = row * D
    cols    = tl.arange(0, BLOCK_SIZE)
    mask    = cols < D

    # Masked loads: padded slots → 0.0  (no tl.where needed for reductions)
    x = tl.load(x_ptr + row_off + cols, mask=mask, other=0.0).to(tl.float32)
    z = tl.load(z_ptr + row_off + cols, mask=mask, other=0.0).to(tl.float32)
    w = tl.load(w_ptr + cols,           mask=mask, other=1.0).to(tl.float32)
    b = tl.load(b_ptr + cols,           mask=mask, other=0.0).to(tl.float32)

    xz   = x + z          # padded slots = 0.0 + 0.0 = 0.0
    xzsq = xz * xz        # padded slots = 0.0 (correct for sum-of-squares)

    # Single combined reduction: Σx and Σx² in ONE synchronisation barrier
    s1, s2 = tl.reduce((xz, xzsq), axis=0, combine_fn=_sum_sq_combine)

    mean = s1 / D
    var  = s2 / D - mean * mean
    rstd = tl.rsqrt(var + eps)

    # Triton auto-casts fp32 → output dtype on store
    out = (xz - mean) * rstd * w + b
    tl.store(out_ptr + row_off + cols, out, mask=mask)


# ---------------------------------------------------------------------------
# Python wrapper
# ---------------------------------------------------------------------------

@torch.fx.wrap
def fused_add_layernorm(in_5, in_6, in_1, in_2):
    D  = in_5.shape[-1]     # 384
    N  = in_5.numel() // D  # 578
    out = torch.empty_like(in_5)

    fused_add_layernorm_kernel[(N,)](
        in_5, in_6, in_2, in_1, out,
        D,      # positional: row length
        1e-12,  # positional: eps
        BLOCK_SIZE=512,
        num_warps=4,
    )

    return out.view(in_5.shape)


def replacement_func():
    return fused_add_layernorm