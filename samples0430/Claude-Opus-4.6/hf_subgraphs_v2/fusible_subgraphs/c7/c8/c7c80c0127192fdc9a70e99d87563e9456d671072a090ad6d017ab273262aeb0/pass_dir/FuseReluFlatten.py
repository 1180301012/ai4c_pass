import torch
import triton
import triton.language as tl


def pattern(x):
    tmp_0 = torch.nn.functional.relu(x, inplace=True)
    tmp_1 = tmp_0.flatten(1, -1)
    return tmp_1


def replacement_args(x):
    return (x,)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 256}, num_warps=2),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=8),
    ],
    key=['n_elements'],
)
@triton.jit
def fused_relu_flatten_kernel(
    x_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    # ReLU
    zero = tl.zeros_like(x)
    out = tl.maximum(x, zero)
    tl.store(out_ptr + offsets, out, mask=mask)


@torch.fx.wrap
def fused_relu_flatten(x):
    # Get shape info
    batch_size = x.shape[0]
    # Compute flattened size from dim 1 onwards
    flat_size = 1
    for i in range(1, len(x.shape)):
        flat_size *= x.shape[i]
    
    n_elements = batch_size * flat_size
    
    # Allocate output with flattened shape
    out = torch.empty((batch_size, flat_size), dtype=x.dtype, device=x.device)
    
    # Launch kernel
    grid = lambda meta: ((n_elements + meta['BLOCK_SIZE'] - 1) // meta['BLOCK_SIZE'],)
    fused_relu_flatten_kernel[grid](
        x_ptr=x,
        out_ptr=out,
        n_elements=n_elements,
    )
    
    return out


def replacement_func():
    return fused_relu_flatten