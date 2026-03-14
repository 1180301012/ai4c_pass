import torch
import triton
import triton.language as tl


# Pattern matching function - simple mul
def pattern(in_0, in_1):
    result = in_0 * in_1
    return result


def replacement_args(in_0, in_1):
    return (in_0, in_1)


# Optimized Triton kernel for element-wise multiplication
@triton.jit
def mul_kernel(
    x_ptr,
    y_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a contiguous block of data of size BLOCK_SIZE
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements  # Mask to ensure we don't go out of bounds
    
    # Load x and y
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = tl.load(y_ptr + offsets, mask=mask, other=0.0)
    
    # Compute x * y
    out = x * y
    
    # Store result
    tl.store(out_ptr + offsets, out, mask=mask)


@torch.fx.wrap
def kernel_wrapper(x, y):
    # Get total number of elements
    n_elements = x.numel()
    # Use multiple programs for better parallelism
    BLOCK_SIZE = 2048
    num_programs = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Allocate output tensor
    out = torch.empty_like(x)
    
    # Launch kernel
    mul_kernel[(num_programs,)](
        x_ptr=x,
        y_ptr=y,
        out_ptr=out,
        n_elements=n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out


def replacement_func():
    return kernel_wrapper