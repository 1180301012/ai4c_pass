"""
Fuses:
    tmp_0 = in_1.norm(p=2, dim=-1, keepdim=True)
    tmp_1 = in_1 / tmp_0
into a single Triton L2-normalisation kernel — true single-pass:
loads each row ONCE, computes norm in-register, normalises, stores.

No routing overhead: replacement_args passes the tensor directly (no route
string) since this is the only active pass.
"""

import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Triton kernel
# BLOCK_SIZE=2048 covers all three column widths (768/1024/1152).
# do_not_specialize keeps a single compiled variant → faster dispatch lookup.
# num_warps=16 (512 threads) matches typical PyTorch reduction block size.
# ---------------------------------------------------------------------------

@triton.jit(do_not_specialize=["n_cols", "stride_row"])
def _l2_norm_kernel(
    x_ptr,
    out_ptr,
    n_cols,
    stride_row,
    BLOCK_SIZE: tl.constexpr,
):
    row_idx  = tl.program_id(0)
    row_base = row_idx * stride_row
    cols     = tl.arange(0, BLOCK_SIZE)
    mask     = cols < n_cols

    x      = tl.load(x_ptr + row_base + cols, mask=mask, other=0.0).to(tl.float32)
    sum_sq = tl.sum(x * x, axis=0)
    norm   = tl.sqrt(sum_sq)
    out    = x / norm
    tl.store(out_ptr + row_base + cols, out.to(tl.bfloat16), mask=mask)


# ---------------------------------------------------------------------------
# Wrapper
# ---------------------------------------------------------------------------

@torch.fx.wrap
def triton_l2_normalize(in_1):
    out = torch.empty_like(in_1)
    _l2_norm_kernel[(in_1.shape[0],)](
        in_1, out, in_1.shape[1], in_1.stride(0),
        BLOCK_SIZE=2048, num_warps=32,
    )
    return out


# ---------------------------------------------------------------------------
# Pass interface
# ---------------------------------------------------------------------------

def pattern(in_1):
    tmp_0 = in_1.norm(p=2, dim=-1, keepdim=True)
    tmp_1 = in_1 / tmp_0
    return tmp_1


def replacement_args(in_1):
    return (in_1,)


def replacement_func():
    return triton_l2_normalize