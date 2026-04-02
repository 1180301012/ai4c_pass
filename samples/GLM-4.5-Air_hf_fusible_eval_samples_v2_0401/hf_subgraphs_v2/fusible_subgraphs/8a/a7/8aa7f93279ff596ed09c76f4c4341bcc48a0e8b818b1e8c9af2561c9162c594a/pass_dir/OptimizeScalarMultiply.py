import torch
import triton
import triton.language as tl
import math

def pattern(in_1):
    tmp_0 = in_1 * 0.1767766952966369
    return tmp_0

def replacement_args(in_1):
    return (in_1,)

@triton.jit
def scalar_multiply_kernel_fp16(
    x_ptr,
    out_ptr,
    scalar,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a contiguous block of data
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements  # Mask to ensure we don't go out of bounds
    
    # Load input data as float16
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    
    # Perform multiplication with scalar directly in fp16
    out = x * scalar
    
    # Store result
    tl.store(out_ptr + offsets, out, mask=mask)

@triton.jit
def scalar_multiply_kernel_bf16(
    x_ptr,
    out_ptr,
    scalar,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a contiguous block of data
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements  # Mask to ensure we don't go out of bounds
    
    # Load input data as bfloat16
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    
    # Perform multiplication with scalar directly in bf16
    out = x * scalar
    
    # Store result
    tl.store(out_ptr + offsets, out, mask=mask)

@triton.jit
def scalar_multiply_kernel_fp32(
    x_ptr,
    out_ptr,
    scalar,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a contiguous block of data
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements  # Mask to ensure we don't go out of bounds
    
    # Load input data as float32
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    
    # Perform multiplication with scalar directly in fp32
    out = x * scalar
    
    # Store result
    tl.store(out_ptr + offsets, out, mask=mask)

@torch.fx.wrap
def optimized_scalar_multiply(x, scalar_value):
    N = x.numel()
    
    # Optimize BLOCK_SIZE based on tensor size
    if N < 1024:
        BLOCK_SIZE = 256
    elif N < 8192:
        BLOCK_SIZE = 512
    elif N < 65536:
        BLOCK_SIZE = 1024
    else:
        BLOCK_SIZE = 2048
    
    num_programs = (N + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Handle different data types with specific kernels for each type
    if x.dtype == torch.float16:
        result = torch.empty_like(x)
        scalar_multiply_kernel_fp16[(num_programs,)](
            x_ptr=x,
            out_ptr=result,
            scalar=scalar_value,
            n_elements=N,
            BLOCK_SIZE=BLOCK_SIZE,
        )
        return result
    elif x.dtype == torch.bfloat16:
        result = torch.empty_like(x)
        scalar_multiply_kernel_bf16[(num_programs,)](
            x_ptr=x,
            out_ptr=result,
            scalar=scalar_value,
            n_elements=N,
            BLOCK_SIZE=BLOCK_SIZE,
        )
        return result
    else:  # float32
        result = torch.empty_like(x)
        scalar_multiply_kernel_fp32[(num_programs,)](
            x_ptr=x,
            out_ptr=result,
            scalar=scalar_value,
            n_elements=N,
            BLOCK_SIZE=BLOCK_SIZE,
        )
        return result

def replacement_func():
    # This returns a function that takes (in_1,) and returns tmp_0
    def wrapper(in_1):
        return optimized_scalar_multiply(in_1, 0.1767766952966369)
    return wrapper