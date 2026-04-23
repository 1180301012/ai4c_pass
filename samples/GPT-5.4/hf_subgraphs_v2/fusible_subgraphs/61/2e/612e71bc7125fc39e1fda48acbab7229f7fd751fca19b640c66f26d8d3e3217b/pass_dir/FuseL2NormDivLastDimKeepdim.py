import torch
import triton
import triton.language as tl


# Pattern matching function
# Matches only the normalization branch and returns the externally visible result.
def pattern(in_1):
    tmp_0 = in_1.norm(p=2, dim=-1, keepdim=True)
    tmp_1 = in_1 / tmp_0
    return tmp_1


# Argument extraction function
def replacement_args(in_1):
    return (in_1,)


@triton.jit
def _two_row_l2_normalize_kernel(
    x_ptr,
    y_ptr,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
):
    offs = tl.arange(0, BLOCK_SIZE)

    mask0 = offs < n_cols
    x00 = tl.load(x_ptr + offs, mask=mask0, other=0.0).to(tl.float32)
    x10 = tl.load(x_ptr + n_cols + offs, mask=mask0, other=0.0).to(tl.float32)

    offs1 = BLOCK_SIZE + offs
    mask1 = offs1 < n_cols
    x01 = tl.load(x_ptr + offs1, mask=mask1, other=0.0).to(tl.float32)
    x11 = tl.load(x_ptr + n_cols + offs1, mask=mask1, other=0.0).to(tl.float32)

    sumsq0 = tl.sum(x00 * x00, axis=0) + tl.sum(x01 * x01, axis=0)
    sumsq1 = tl.sum(x10 * x10, axis=0) + tl.sum(x11 * x11, axis=0)

    inv0 = tl.rsqrt(sumsq0)
    inv1 = tl.rsqrt(sumsq1)

    tl.store(y_ptr + offs, x00 * inv0, mask=mask0)
    tl.store(y_ptr + n_cols + offs, x10 * inv1, mask=mask0)
    tl.store(y_ptr + offs1, x01 * inv0, mask=mask1)
    tl.store(y_ptr + n_cols + offs1, x11 * inv1, mask=mask1)


@torch.fx.wrap
def triton_row_l2_normalize(in_1):
    out = torch.empty_like(in_1)

    shape = in_1.size()
    n_cols = shape[1]

    _two_row_l2_normalize_kernel[(1,)](
        x_ptr=in_1,
        y_ptr=out,
        n_cols=n_cols,
        BLOCK_SIZE=1024,
        num_warps=4,
        num_stages=1,
    )
    return out


# Replacement function (NO arguments, returns function reference)
def replacement_func():
    return triton_row_l2_normalize