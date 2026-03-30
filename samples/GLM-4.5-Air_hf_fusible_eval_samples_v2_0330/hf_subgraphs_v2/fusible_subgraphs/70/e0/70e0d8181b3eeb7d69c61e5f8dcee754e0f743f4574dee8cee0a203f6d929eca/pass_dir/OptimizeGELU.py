import torch
import triton
import triton.language as tl
import math

@triton.jit
def gelu_kernel(
    input_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a contiguous block of data
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # Mask to ensure we don't go out of bounds
    mask = offsets < n_elements
    
    # Load input values
    x = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    
    # GELU approximation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    # This is the same approximation used by PyTorch
    x_cubed = x * x * x
    inner = x + 0.044715 * x_cubed
    approx_tanh = math.tanh(math.sqrt(2.0 / math.pi) * inner)
    gelu_val = 0.5 * x * (1.0 + approx_tanh)
    
    # Store result
    tl.store(output_ptr + offsets, gelu_val, mask=mask)

@torch.fx.wrap
def optimized_gelu(x):
    """Optimized GELU using Triton kernel"""
    n_elements = x.numel()
    BLOCK_SIZE = 1024  # Standard block size for good GPU occupancy
    num_programs = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    output = torch.empty_like(x)
    
    gelu_kernel[(num_programs,)](
        input_ptr=x,
        output_ptr=output,
        n_elements=n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return output

def pattern(input_tensor):
    """
    Pattern that matches GELU computation
    """
    return torch.nn.functional.gelu(input_tensor)

def replacement_args(input_tensor):
    """Extract arguments for the replacement function"""
    return (input_tensor,)

def replacement_func():
    """Return the optimized GELU function"""
    return optimized_gelu