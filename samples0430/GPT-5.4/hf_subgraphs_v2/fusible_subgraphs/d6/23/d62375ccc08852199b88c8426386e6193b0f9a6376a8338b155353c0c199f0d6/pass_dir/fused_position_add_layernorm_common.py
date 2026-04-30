import torch
import triton
import triton.language as tl


@triton.jit
def _layernorm_kernel(
    x_ptr,
    weight_ptr,
    bias_ptr,
    y_ptr,
    hidden,
    eps,
    x_row_stride,
    x_col_stride,
    y_row_stride,
    y_col_stride,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < hidden

    offsets = row * x_row_stride + cols * x_col_stride
    x = tl.load(x_ptr + offsets, mask=mask, other=0)
    x_f32 = x.to(tl.float32)
    mean = tl.sum(x_f32, axis=0) / hidden
    centered = x_f32 - mean
    var = tl.sum(centered * centered, axis=0) / hidden
    rstd = tl.rsqrt(var + eps)

    weight = tl.load(weight_ptr + cols, mask=mask, other=0).to(tl.float32)
    bias = tl.load(bias_ptr + cols, mask=mask, other=0).to(tl.float32)
    y = centered * rstd * weight + bias

    tl.store(y_ptr + row * y_row_stride + cols * y_col_stride, y, mask=mask)


@torch.fx.wrap
def triton_layernorm(x, bias, weight):
    y = torch.empty_like(x)
    rows = x.shape[1]
    hidden = x.shape[2]
    block_size = triton.next_power_of_2(hidden)
    num_warps = 4 if hidden <= 1024 else 8

    _layernorm_kernel[(rows,)](
        x,
        weight,
        bias,
        y,
        hidden,
        1e-5,
        x.stride(1),
        x.stride(2),
        y.stride(1),
        y.stride(2),
        BLOCK_SIZE=block_size,
        num_warps=num_warps,
        num_stages=1,
    )
    return y


def replacement_func():
    return triton_layernorm