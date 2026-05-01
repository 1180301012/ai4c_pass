import torch
import triton
import triton.language as tl


def pattern(x, weight, bias):
    channel_dim = x.shape[-1]
    return torch.nn.functional.layer_norm(x, (channel_dim,), weight, bias, 1e-05)

def replacement_args(x, weight, bias):
    return (x, weight, bias)


@triton.jit
def layer_norm_kernel(
    x_ptr,
    weight_ptr,
    bias_ptr,
    out_ptr,
    H,
    C,
    eps,
    BLOCK_SIZE_C: tl.constexpr,
):
    h = tl.program_id(0)
    if h >= H:
        return

    sum_x = 0.0
    sum_x2 = 0.0
    for c in range(0, C, BLOCK_SIZE_C):
        c_start = c
        c_end = min(C, c + BLOCK_SIZE_C)
        num_c = c_end - c_start
        x_block = tl.load(x_ptr + h * C + c_start, mask=tl.arange(0, num_c) < num_c, other=0.0)
        sum_x += tl.sum(x_block)
        sum_x2 += tl.sum(x_block * x_block)

    mean = sum_x / C
    var = sum_x2 / C - mean * mean
    inv_std = 1.0 / tl.sqrt(var + eps)

    for c in range(0, C, BLOCK_SIZE_C):
        c_start = c
        c_end = min(C, c + BLOCK_SIZE_C)
        num_c = c_end - c_start
        x_block = tl.load(x_ptr + h * C + c_start, mask=tl.arange(0, num_c) < num_c, other=0.0)
        normalized = (x_block - mean) * inv_std
        weight_block = tl.load(weight_ptr + c_start, mask=tl.arange(0, num_c) < num_c, other=0.0)
        bias_block = tl.load(bias_ptr + c_start, mask=tl.arange(0, num_c) < num_c, other=0.0)
        out_block = normalized * weight_block + bias_block
        tl.store(out_ptr + h * C + c_start, out_block, mask=tl.arange(0, num_c) < num_c)


@torch.fx.wrap
def layer_norm_wrapper(x, weight, bias, eps=1e-05):
    B, H, C = x.shape
    out = torch.empty_like(x)
    BLOCK_SIZE_C = 128
    grid = (H,)
    layer_norm_kernel[grid](
        x_ptr=x,
        weight_ptr=weight,
        bias_ptr=bias,
        out_ptr=out,
        H=H,
        C=C,
        eps=eps,
        BLOCK_SIZE_C=BLOCK_SIZE_C,
    )
    return out

def replacement_func():
    return layer_norm_wrapper