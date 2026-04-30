import torch
import triton
import triton.language as tl


@triton.jit
def _relu_sigmoid_kernel(
    x_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    x_relu = tl.maximum(x, 0)
    tl.store(x_ptr + offsets, x_relu, mask=mask)

    x_f32 = x_relu.to(tl.float32)
    y = 1.0 / (1.0 + tl.exp(-x_f32))
    tl.store(out_ptr + offsets, y, mask=mask)


@triton.jit
def _sigmoid_inplace_kernel(
    x_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    x_f32 = x.to(tl.float32)
    y = 1.0 / (1.0 + tl.exp(-x_f32))
    tl.store(x_ptr + offsets, y, mask=mask)


@torch.fx.wrap
def shared_activation_dispatch(x, route: str):
    n_elements = x.numel()

    if n_elements <= 1024:
        block_size = 1024
        num_warps = 4
    elif n_elements <= 4096:
        block_size = 1024
        num_warps = 4
    else:
        block_size = 2048
        num_warps = 8

    grid = (triton.cdiv(n_elements, block_size),)

    if route == "relu_sigmoid":
        out = torch.empty_like(x)
        _relu_sigmoid_kernel[grid](
            x,
            out,
            n_elements,
            BLOCK_SIZE=block_size,
            num_warps=num_warps,
            num_stages=1,
        )
        return out
    elif route == "sigmoid":
        _sigmoid_inplace_kernel[grid](
            x,
            n_elements,
            BLOCK_SIZE=block_size,
            num_warps=num_warps,
            num_stages=1,
        )
        return x
    else:
        raise ValueError(f"Unknown route: {route}")