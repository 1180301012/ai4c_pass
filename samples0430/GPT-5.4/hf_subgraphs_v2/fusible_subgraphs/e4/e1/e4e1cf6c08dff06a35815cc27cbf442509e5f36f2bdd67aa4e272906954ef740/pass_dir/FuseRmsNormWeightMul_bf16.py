import torch
import triton
import triton.language as tl


def pattern(tmp_2, in_1):
    tmp_4 = tmp_2.float()
    tmp_5 = tmp_4.pow(2)
    tmp_6 = tmp_5.mean(-1, keepdim=True)
    tmp_7 = tmp_6 + 1e-06
    tmp_8 = torch.rsqrt(tmp_7)
    tmp_9 = tmp_4 * tmp_8
    tmp_10 = in_1.float()
    tmp_11 = 1.0 + tmp_10
    tmp_12 = tmp_9 * tmp_11
    tmp_13 = tmp_12.type_as(tmp_2)
    return tmp_13


def replacement_args(tmp_2, in_1):
    return (tmp_2, in_1)


@triton.jit
def _rmsnorm_weightmul_kernel(
    x_ptr,
    w_ptr,
    out_ptr,
    n_rows,
    n_cols,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    if row >= n_rows:
        return

    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < n_cols
    row_start = row * n_cols

    x = tl.load(x_ptr + row_start + cols, mask=mask, other=0).to(tl.float32)
    w = tl.load(w_ptr + cols, mask=mask, other=0).to(tl.float32)

    mean_sq = tl.sum(x * x, axis=0) / n_cols
    inv_rms = tl.rsqrt(mean_sq + eps)
    y = x * inv_rms * (1.0 + w)

    tl.store(out_ptr + row_start + cols, y.to(tl.bfloat16), mask=mask)


@torch.fx.wrap
def _rmsnorm_weightmul_triton(tmp_2, in_1):
    out = torch.empty_like(tmp_2)

    n_cols = tmp_2.shape[-1]
    n_rows = tmp_2.numel() // n_cols

    BLOCK_SIZE = 2048

    _rmsnorm_weightmul_kernel[(n_rows,)](
        tmp_2,
        in_1,
        out,
        n_rows,
        n_cols,
        1e-06,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=8,
        num_stages=2,
    )
    return out


def replacement_func():
    return _rmsnorm_weightmul_triton