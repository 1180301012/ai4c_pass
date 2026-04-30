import torch
import triton
import triton.language as tl


def pattern(x, y):
    # Simplest possible test: does the graph use operator.mul?
    return x * y


def replacement_args(x, y):
    return (x, y)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 8192}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 16384}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 32768}, num_warps=8),
    ],
    key=['n_elements'],
)
@triton.jit
def silu_mul_kernel(
    x_ptr,
    y_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = tl.load(y_ptr + offsets, mask=mask, other=0.0)

    # SiLU: x * sigmoid(x)
    silu_x = x * tl.sigmoid(x)

    # Multiply with y (dropout p=0.0 is identity)
    out = silu_x * y

    tl.store(out_ptr + offsets, out, mask=mask)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 8192}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 16384}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 32768}, num_warps=8),
    ],
    key=['n_elements'],
)
@triton.jit
def mul_only_kernel(
    x_ptr,
    y_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = tl.load(y_ptr + offsets, mask=mask, other=0.0)

    out = x * y

    tl.store(out_ptr + offsets, out, mask=mask)


@torch.fx.wrap
def fused_silu_mul_dropout(x, y):
    N = x.numel()
    out = torch.empty_like(x)
    grid = lambda meta: ((N + meta['BLOCK_SIZE'] - 1) // meta['BLOCK_SIZE'],)
    mul_only_kernel[grid](
        x,
        y,
        out,
        N,
    )
    return out


def replacement_func():
    return fused_silu_mul_dropout