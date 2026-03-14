import torch
import triton
import triton.language as tl
import math

def pattern(in_tensor):
    result = in_tensor.contiguous()
    return result

def replacement_args(in_tensor):
    return (in_tensor,)

@triton.jit
def optimized_contiguous_kernel(
    input_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a contiguous block of data
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offset = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offset < n_elements
    
    # Load from input and store to output
    # This ensures memory alignment and contiguous layout
    input_data = tl.load(input_ptr + offset, mask=mask, other=0.0)
    tl.store(output_ptr + offset, input_data, mask=mask)

@triton.jit
def optimized_contiguous_kernel(
    input_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a contiguous block of data
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offset = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offset < n_elements
    
    # Load from input and store to output
    # This ensures memory alignment and contiguous layout
    input_data = tl.load(input_ptr + offset, mask=mask, other=0.0)
    tl.store(output_ptr + offset, input_data, mask=mask)

@torch.fx.wrap
def optimized_contiguous(in_tensor):
    n_elements = in_tensor.numel()
    
    # If tensor is already contiguous, return as-is to avoid unnecessary copy
    if in_tensor.is_contiguous():
        return in_tensor
    
    # For very small tensors, use PyTorch's native contiguous (faster than kernel launch)
    if n_elements < 4096:
        return in_tensor.contiguous()
    
    # Select optimal block size based on tensor size
    if n_elements < 100_000:
        BLOCK_SIZE = 512
    elif n_elements < 1_000_000:
        BLOCK_SIZE = 1024
    else:
        BLOCK_SIZE = 2048
    
    # Create output tensor
    output = torch.empty_like(in_tensor)
    
    # Calculate grid dimensions with slight over-subscription for better GPU utilization
    num_programs = math.ceil(n_elements / BLOCK_SIZE)
    
    # Launch autotuned kernel
    optimized_contiguous_kernel[(num_programs,)](
        input_ptr=in_tensor,
        output_ptr=output,
        n_elements=n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return output

def replacement_func():
    return optimized_contiguous