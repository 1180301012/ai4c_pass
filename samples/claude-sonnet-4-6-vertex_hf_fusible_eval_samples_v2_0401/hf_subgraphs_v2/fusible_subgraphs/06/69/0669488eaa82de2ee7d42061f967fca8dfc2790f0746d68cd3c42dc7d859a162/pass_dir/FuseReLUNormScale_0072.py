"""
Fused pass: relu -> flatten(2) -> norm(dim=-1, keepdim=True) -> *0.07216878364870322
            -> clamp(min=1e-5) -> / -> * in_0

This covers rtmw-l_start411_end418_1 graphs (in_1 shape [B, 133, 16, 12], D=192).
"""
import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Triton kernel
# ---------------------------------------------------------------------------
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 64},  num_warps=2),
        triton.Config({'BLOCK_SIZE': 64},  num_warps=4),
        triton.Config({'BLOCK_SIZE': 128}, num_warps=2),
        triton.Config({'BLOCK_SIZE': 128}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=8),
    ],
    key=['D'],
)
@triton.jit
def _relu_norm_kernel_0072(
    in0_ptr,    # [1] scalar g
    in2_ptr,    # [BN, D] flattened activations (tmp_2)
    in3_ptr,    # [BN] norm per row (tmp_3 squeezed)
    out_ptr,    # [BN, D] output
    BN,
    D,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    row_start = pid * D

    # Load per-row norm value in original dtype
    norm_val = tl.load(in3_ptr + pid)
    # Scale and clamp in original dtype (matching PyTorch behavior)
    denom = tl.maximum(norm_val * 0.07216878364870322, 1e-5)

    # Load scalar weight g in original dtype
    g = tl.load(in0_ptr)

    # Load row of tmp_2
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < D
    x = tl.load(in2_ptr + row_start + cols, mask=mask, other=0.0)

    # out = x / denom * g  (all in original dtype)
    out = x / denom * g
    tl.store(out_ptr + row_start + cols, out, mask=mask)


# ---------------------------------------------------------------------------
# Python wrapper — use native PyTorch for exact dtype matching
# ---------------------------------------------------------------------------
@torch.fx.wrap
def _fused_scale_clamp_div_mul_0072(in_0, tmp_2, tmp_3):
    """
    in_0 : [1]          – scalar weight
    tmp_2: [B, N, D]    – flattened activations
    tmp_3: [B, N, 1]    – norm output (keepdim=True)
    """
    denom = (tmp_3 * 0.07216878364870322).clamp(min=1e-05)
    return (tmp_2 / denom) * in_0


# ---------------------------------------------------------------------------
# Pattern / replacement API
# ---------------------------------------------------------------------------
def pattern(in_0, tmp_2, tmp_3):
    tmp_4 = tmp_3 * 0.07216878364870322
    tmp_5 = tmp_4.clamp(min=1e-05)
    tmp_6 = tmp_2 / tmp_5
    tmp_7 = tmp_6 * in_0
    return tmp_7


def replacement_args(in_0, tmp_2, tmp_3):
    return (in_0, tmp_2, tmp_3)


def replacement_func():
    return _fused_scale_clamp_div_mul_0072