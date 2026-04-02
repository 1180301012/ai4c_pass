"""
Fused pass for post-norm ops:
  tmp_4 = tmp_3 * 0.14433756729740643
  tmp_5 = clamp(tmp_4, min=1e-5)
  tmp_6 = tmp_2 / tmp_5
  tmp_7 = tmp_6 * in_0

Covers rtmw-l_start396_end403_0 (scale=0.14433756729740643).
"""
import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Triton kernel: scale+clamp+div+mul (norm already computed externally)
# ---------------------------------------------------------------------------
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 64},  num_warps=2),
        triton.Config({'BLOCK_SIZE': 64},  num_warps=4),
        triton.Config({'BLOCK_SIZE': 128}, num_warps=2),
        triton.Config({'BLOCK_SIZE': 128}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=8),
    ],
    key=['D'],
)
@triton.jit
def _scale_clamp_div_mul_kernel_0144(
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
    denom = tl.maximum(norm_val * 0.14433756729740643, 1e-5)

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
def _fused_scale_clamp_div_mul_0144(in_0, tmp_2, tmp_3):
    """
    in_0 : [1]          – scalar weight
    tmp_2: [B, N, D]    – flattened activations
    tmp_3: [B, N, 1]    – norm output (keepdim=True)
    Fuse: scale → clamp → div → mul into one call.
    """
    denom = (tmp_3 * 0.14433756729740643).clamp(min=1e-05)
    return (tmp_2 / denom) * in_0


# ---------------------------------------------------------------------------
# Pattern / replacement API
# ---------------------------------------------------------------------------
def pattern(in_0, tmp_2, tmp_3):
    """
    Match the 4 ops after norm:
      tmp_4 = tmp_3 * scale
      tmp_5 = clamp(tmp_4, min=1e-5)
      tmp_6 = tmp_2 / tmp_5
      tmp_7 = tmp_6 * in_0
    """
    tmp_4 = tmp_3 * 0.14433756729740643
    tmp_5 = tmp_4.clamp(min=1e-05)
    tmp_6 = tmp_2 / tmp_5
    tmp_7 = tmp_6 * in_0
    return tmp_7


def replacement_args(in_0, tmp_2, tmp_3):
    return (in_0, tmp_2, tmp_3)


def replacement_func():
    return _fused_scale_clamp_div_mul_0144