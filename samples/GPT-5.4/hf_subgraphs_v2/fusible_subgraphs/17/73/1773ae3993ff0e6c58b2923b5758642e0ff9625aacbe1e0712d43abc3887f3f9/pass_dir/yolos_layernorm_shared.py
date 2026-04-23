import torch
import triton
import triton.language as tl


@triton.jit
def _layer_norm_kernel(
    x_ptr,
    weight_ptr,
    bias_ptr,
    out_ptr,
    rows,
    n_cols,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < n_cols
    row_start = row * n_cols

    x = tl.load(x_ptr + row_start + cols, mask=mask, other=0.0).to(tl.float32)
    mean = tl.sum(x, axis=0) / n_cols
    x_centered = x - mean
    var = tl.sum(x_centered * x_centered, axis=0) / n_cols
    rstd = tl.rsqrt(var + eps)

    weight = tl.load(weight_ptr + cols, mask=mask, other=1.0).to(tl.float32)
    bias = tl.load(bias_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    y = x_centered * rstd
    y = y * weight + bias
    tl.store(out_ptr + row_start + cols, y, mask=mask)


def _select_block_size(n_cols):
    if n_cols <= 32:
        return 32
    if n_cols <= 64:
        return 64
    if n_cols <= 128:
        return 128
    if n_cols <= 256:
        return 256
    if n_cols <= 512:
        return 512
    return 1024


def _select_num_warps(n_cols):
    if n_cols <= 64:
        return 1
    if n_cols <= 256:
        return 4
    return 8


@torch.fx.wrap
def yolos_layer_norm(x, weight, bias):
    n_cols = weight.numel()
    rows = x.numel() // n_cols
    out = torch.empty_like(x)

    block_size = _select_block_size(n_cols)
    num_warps = _select_num_warps(n_cols)

    _layer_norm_kernel[(rows,)](
        x,
        weight,
        bias,
        out,
        rows,
        n_cols,
        1e-12,
        BLOCK_SIZE=block_size,
        num_warps=num_warps,
        num_stages=1,
    )
    return out


def yolos_layer_norm_replacement_args(x, weight, bias):
    return (x, weight, bias)


def yolos_layer_norm_replacement_func():
    return yolos_layer_norm