import torch
import triton
import triton.language as tl


def pattern(in_0, in_1, in_2, in_3, in_4):
    tmp_5 = torch.nn.functional.batch_norm(in_4, in_0, in_1, in_3, in_2, False, 0.1, 1e-05)
    return tmp_5


def replacement_args(in_0, in_1, in_2, in_3, in_4):
    return (in_0, in_1, in_2, in_3, in_4)


@triton.jit
def batch_norm_inference_kernel(
    x_ptr,
    mean_ptr,
    var_ptr,
    bias_ptr,
    weight_ptr,
    out_ptr,
    m,
    c,
    x_stride0,
    x_stride1,
    out_stride0,
    out_stride1,
    eps,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)

    rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = tl.arange(0, BLOCK_N)

    row_mask = rows < m
    col_mask = cols < c
    mask = row_mask[:, None] & col_mask[None, :]

    x_ptrs = x_ptr + rows[:, None] * x_stride0 + cols[None, :] * x_stride1
    x = tl.load(x_ptrs, mask=mask, other=0.0)

    mean = tl.load(mean_ptr + cols, mask=col_mask, other=0.0).to(tl.float32)
    var = tl.load(var_ptr + cols, mask=col_mask, other=1.0).to(tl.float32)
    bias = tl.load(bias_ptr + cols, mask=col_mask, other=0.0).to(tl.float32)
    weight = tl.load(weight_ptr + cols, mask=col_mask, other=1.0).to(tl.float32)

    x = x.to(tl.float32)
    y = ((x - mean[None, :]) * tl.rsqrt(var[None, :] + eps)) * weight[None, :] + bias[None, :]

    out_ptrs = out_ptr + rows[:, None] * out_stride0 + cols[None, :] * out_stride1
    tl.store(out_ptrs, y, mask=mask)


@torch.fx.wrap
def batch_norm_inference(mean, var, bias, weight, x):
    out = torch.empty_like(x)

    m = x.shape[0]
    c = x.shape[1]

    grid = (triton.cdiv(m, 8),)
    batch_norm_inference_kernel[grid](
        x,
        mean,
        var,
        bias,
        weight,
        out,
        m,
        c,
        x.stride(0),
        x.stride(1),
        out.stride(0),
        out.stride(1),
        1e-5,
        BLOCK_M=8,
        BLOCK_N=128,
    )
    return out


def replacement_func():
    return batch_norm_inference