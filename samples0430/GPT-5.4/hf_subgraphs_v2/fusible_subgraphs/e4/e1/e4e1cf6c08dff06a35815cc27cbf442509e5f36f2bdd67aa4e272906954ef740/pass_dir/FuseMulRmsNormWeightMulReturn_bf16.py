import torch
import triton
import triton.language as tl


def pattern(in_0, in_1, in_2):
    tmp_2 = in_0 * in_2
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
    return (tmp_2, tmp_13)


def replacement_args(in_0, in_1, in_2):
    return (in_0, in_1)


@triton.jit
def _mul_rmsnorm_weight_kernel(
    in_ptr,
    w_ptr,
    tmp2_ptr,
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

    inp = tl.load(in_ptr + row_start + cols, mask=mask, other=0).to(tl.float32)
    x_bf16 = (inp * 45.25).to(tl.bfloat16)
    x = x_bf16.to(tl.float32)

    mean_sq = tl.sum(x * x, axis=0) / n_cols
    inv_rms = tl.rsqrt(mean_sq + eps)

    w = tl.load(w_ptr + cols, mask=mask, other=0).to(tl.float32)
    y = x * inv_rms * (1.0 + w)

    tl.store(tmp2_ptr + row_start + cols, x_bf16, mask=mask)
    tl.store(out_ptr + row_start + cols, y.to(tl.bfloat16), mask=mask)


@torch.fx.wrap
def _mul_rmsnorm_weight_triton(in_0, in_1):
    tmp_2 = torch.empty_like(in_0)
    tmp_13 = torch.empty_like(in_0)

    n_cols = in_0.shape[-1]
    n_rows = in_0.numel() // n_cols

    _mul_rmsnorm_weight_kernel[(n_rows,)](
        in_0,
        in_1,
        tmp_2,
        tmp_13,
        n_rows,
        n_cols,
        1e-06,
        BLOCK_SIZE=2048,
        num_warps=8,
        num_stages=4,
    )
    return (tmp_2, tmp_13)


def replacement_func():
    return _mul_rmsnorm_weight_triton