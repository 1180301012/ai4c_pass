import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Pattern: x.sum(dim=3, keepdim=True) -> x / sum
#
# Shape: x = [1, 2, 8, 8]
#   sum over dim=3 (W=8) → [1, 2, 8, 1]
#   x / sum                → [1, 2, 8, 8]
#
# Interpretation: 16 independent rows of 8 elements each.
#   For row r: out[r, w] = x[r, w] / sum_w'(x[r, w'])
# ---------------------------------------------------------------------------


def pattern(x):
    s = x.sum(dim=3, keepdim=True)
    out = x / s
    return out


def replacement_args(x):
    # Pad to 3 tensor args + route string so dispatch signature matches
    return (x, x, x, "sum_div")


# ---------------------------------------------------------------------------
# Triton kernel
# One program per row (B*C*H = 16 rows), each row has W=8 elements.
# ---------------------------------------------------------------------------

@triton.jit
def _sum_div_dim3_kernel(
    x_ptr,
    out_ptr,
    W:        tl.constexpr,   # 8  – elements per row
    BLOCK_W:  tl.constexpr,   # power-of-2 >= W (= 8)
    IS_BF16:  tl.constexpr,
):
    row_idx   = tl.program_id(0)
    row_start = row_idx * W
    w_offs    = tl.arange(0, BLOCK_W)
    mask      = w_offs < W

    # Load row, accumulate sum in fp32
    x_vals = tl.load(x_ptr + row_start + w_offs, mask=mask, other=0.0).to(tl.float32)
    row_sum = tl.sum(x_vals, axis=0)     # scalar

    # Normalize
    result_f32 = x_vals / row_sum

    # Cast back to input dtype
    if IS_BF16:
        result = result_f32.to(tl.bfloat16)
    else:
        result = result_f32.to(tl.float16)

    tl.store(out_ptr + row_start + w_offs, result, mask=mask)


# ---------------------------------------------------------------------------
# Wrapper
# ---------------------------------------------------------------------------

@torch.fx.wrap
def sum_div_dim3(x):
    """
    Drop-in replacement for:
        s   = x.sum(dim=3, keepdim=True)
        out = x / s
    where x has shape [1, 2, 8, 8].
    """
    B, C, H, W = x.shape      # 1, 2, 8, 8
    N_ROWS = B * C * H         # 16

    out     = torch.empty_like(x)
    is_bf16 = (x.dtype == torch.bfloat16)

    _sum_div_dim3_kernel[(N_ROWS,)](
        x.contiguous(),
        out,
        W=W,
        BLOCK_W=8,   # constexpr: next power-of-2 >= 8
        IS_BF16=is_bf16,
        num_warps=1,
    )

    return out


def replacement_func():
    from pass_dir.shared_dispatch import _dispatch
    return _dispatch