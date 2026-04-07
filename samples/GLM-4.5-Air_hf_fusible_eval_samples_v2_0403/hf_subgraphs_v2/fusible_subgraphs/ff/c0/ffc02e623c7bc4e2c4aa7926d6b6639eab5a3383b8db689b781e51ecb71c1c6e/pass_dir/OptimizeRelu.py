import torch
import triton
import triton.language as tl

def pattern(x):
    """Match ReLU with inplace=True"""
    return torch.nn.functional.relu(x, inplace=True)

def replacement_args(x):
    return (x,)

@triton.jit
def relu_kernel(
    input_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr
):
    # Each program processes a contiguous block of data
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load input
    x = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    
    # Apply ReLU using efficient triton operation
    result = tl.maximum(x, 0.0)
    
    # Store result
    tl.store(output_ptr + offsets, result, mask=mask)

@torch.fx.wrap
def optimized_relu(x):
    n_elements = x.numel()
    BLOCK_SIZE = 1024
    num_programs = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    output = torch.empty_like(x, dtype=x.dtype)
    
    # Ensure contiguous memory access for optimal Triton performance
    x_contiguous = x.contiguous()
    output_contiguous = output.contiguous()
    
    relu_kernel[(num_programs,)](
        x_contiguous,
        output_contiguous,
        n_elements,
        BLOCK_SIZE
    )
    
    return output

def replacement_func():
    return optimized_relu