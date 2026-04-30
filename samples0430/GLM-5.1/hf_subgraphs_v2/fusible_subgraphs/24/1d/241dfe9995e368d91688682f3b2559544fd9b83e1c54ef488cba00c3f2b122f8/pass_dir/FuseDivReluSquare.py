import torch
import triton
import triton.language as tl


def pattern(in_0):
    tmp_0 = in_0 / 11.313708498984761
    tmp_1 = torch.nn.functional.relu(tmp_0)
    tmp_2 = torch.square(tmp_1)
    return tmp_2


def replacement_args(in_0):
    return (in_0,)


@triton.jit
def fused_div_relu_square_kernel(
    x_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    # Precomputed inverse of 11.313708498984761 to use multiplication instead of division
    inv_scale = 0.08838834764831845
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    # Fused: div by scale -> relu -> square
    result = x * inv_scale
    result = tl.maximum(result, 0.0)
    result = result * result
    tl.store(out_ptr + offsets, result, mask=mask)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 256}, num_warps=2),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=2),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=8),
    ],
    key=['n_elements'],
)
@triton.jit
def fused_div_relu_square_kernel_autotuned(
    x_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    inv_scale = 0.08838834764831845
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    result = x * inv_scale
    result = tl.maximum(result, 0.0)
    result = result * result
    tl.store(out_ptr + offsets, result, mask=mask)


@torch.fx.wrap
def fused_div_relu_square(x):
    n_elements = x.numel()
    out = torch.empty_like(x)
    # For small tensors, use a simple kernel; for larger ones, autotuned
    if n_elements < 2048:
        BLOCK_SIZE = 256
        num_programs = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
        fused_div_relu_square_kernel[(num_programs,)](
            x_ptr=x,
            out_ptr=out,
            n_elements=n_elements,
            BLOCK_SIZE=BLOCK_SIZE,
        )
    else:
        fused_div_relu_square_kernel_autotuned[(n_elements,)](
            x_ptr=x,
            out_ptr=out,
            n_elements=n_elements,
        )
    return out


def replacement_func():
    return fused_div_relu_square