import torch
import triton
import triton.language as tl


# D=32 kernel: exact fit with BLOCK_D=32, no masking needed
@triton.jit
def _layer_norm_kernel_32(
    Y_ptr, X_ptr, W_ptr, B_ptr,
    stride_y, stride_x,
    N, D,
    eps: tl.constexpr,
):
    row_idx = tl.program_id(0)
    x_ptr = X_ptr + row_idx * stride_x
    y_ptr = Y_ptr + row_idx * stride_y

    BLOCK_D: tl.constexpr = 32
    offsets = tl.arange(0, BLOCK_D)

    x = tl.load(x_ptr + offsets).to(tl.float32)
    w = tl.load(W_ptr + offsets).to(tl.float32)
    b = tl.load(B_ptr + offsets).to(tl.float32)

    mean = tl.sum(x, axis=0) / D
    x_centered = x - mean
    var = tl.sum(x_centered * x_centered, axis=0) / D
    rstd = 1.0 / tl.sqrt(var + eps)
    x_norm = x_centered * rstd
    y = x_norm * w + b

    tl.store(y_ptr + offsets, y)


@triton.autotune(
    configs=[
        triton.Config({}, num_warps=4, num_stages=2),
        triton.Config({}, num_warps=4, num_stages=3),
        triton.Config({}, num_warps=8, num_stages=2),
        triton.Config({}, num_warps=8, num_stages=3),
        triton.Config({}, num_warps=16, num_stages=2),
        triton.Config({}, num_warps=16, num_stages=3),
    ],
    key=['N', 'D'],
)
@triton.jit
def _layer_norm_kernel_384(
    Y_ptr, X_ptr, W_ptr, B_ptr,
    stride_y, stride_x,
    N, D,
    eps: tl.constexpr,
):
    row_idx = tl.program_id(0)
    x_ptr = X_ptr + row_idx * stride_x
    y_ptr = Y_ptr + row_idx * stride_y

    BLOCK_D: tl.constexpr = 512
    offsets = tl.arange(0, BLOCK_D)
    mask = offsets < D

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    
    x_valid = tl.where(mask, x, 0.0)
    mean = tl.sum(x_valid, axis=0) / D
    
    x_centered_valid = tl.where(mask, x - mean, 0.0)
    var = tl.sum(x_centered_valid * x_centered_valid, axis=0) / D
    rstd = 1.0 / tl.sqrt(var + eps)

    x_norm = tl.where(mask, (x - mean) * rstd, 0.0)

    w = tl.load(W_ptr + offsets, mask=mask, other=1.0).to(tl.float32)
    b = tl.load(B_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    y = x_norm * w + b

    tl.store(y_ptr + offsets, y, mask=mask)


@triton.autotune(
    configs=[
        triton.Config({}, num_warps=8, num_stages=2),
        triton.Config({}, num_warps=8, num_stages=3),
        triton.Config({}, num_warps=16, num_stages=2),
        triton.Config({}, num_warps=16, num_stages=3),
    ],
    key=['N', 'D'],
)
@triton.jit
def _layer_norm_kernel_768(
    Y_ptr, X_ptr, W_ptr, B_ptr,
    stride_y, stride_x,
    N, D,
    eps: tl.constexpr,
):
    row_idx = tl.program_id(0)
    x_ptr = X_ptr + row_idx * stride_x
    y_ptr = Y_ptr + row_idx * stride_y

    BLOCK_D: tl.constexpr = 1024
    offsets = tl.arange(0, BLOCK_D)
    mask = offsets < D

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    
    x_valid = tl.where(mask, x, 0.0)
    mean = tl.sum(x_valid, axis=0) / D
    
    x_centered_valid = tl.where(mask, x - mean, 0.0)
    var = tl.sum(x_centered_valid * x_centered_valid, axis=0) / D
    rstd = 1.0 / tl.sqrt(var + eps)

    x_norm = tl.where(mask, (x - mean) * rstd, 0.0)

    w = tl.load(W_ptr + offsets, mask=mask, other=1.0).to(tl.float32)
    b = tl.load(B_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    y = x_norm * w + b

    tl.store(y_ptr + offsets, y, mask=mask)


def _layer_norm_32_impl(input, weight, bias):
    N = input.numel() // input.shape[-1]
    D = input.shape[-1]
    y = torch.empty_like(input)
    stride_x = input.stride(-2)
    stride_y = y.stride(-2)
    grid = (N,)
    _layer_norm_kernel_32[grid](
        y, input, weight, bias,
        stride_y, stride_x,
        N, D,
        eps=1e-12,
    )
    return y


def _layer_norm_384_impl(input, weight, bias):
    N = input.numel() // input.shape[-1]
    D = input.shape[-1]
    y = torch.empty_like(input)
    stride_x = input.stride(-2)
    stride_y = y.stride(-2)
    grid = (N,)
    _layer_norm_kernel_384[grid](
        y, input, weight, bias,
        stride_y, stride_x,
        N, D,
        eps=1e-12,
    )
    return y


def _layer_norm_768_impl(input, weight, bias):
    N = input.numel() // input.shape[-1]
    D = input.shape[-1]
    y = torch.empty_like(input)
    stride_x = input.stride(-2)
    stride_y = y.stride(-2)
    grid = (N,)
    _layer_norm_kernel_768[grid](
        y, input, weight, bias,
        stride_y, stride_x,
        N, D,
        eps=1e-12,
    )
    return y


@torch.fx.wrap
def layer_norm_dispatch(input, weight, bias, route):
    if route == "ln384":
        return _layer_norm_384_impl(input, weight, bias)
    elif route == "ln32":
        return _layer_norm_32_impl(input, weight, bias)
    elif route == "ln768":
        return _layer_norm_768_impl(input, weight, bias)
    else:
        raise ValueError(f"Unknown route: {route}")