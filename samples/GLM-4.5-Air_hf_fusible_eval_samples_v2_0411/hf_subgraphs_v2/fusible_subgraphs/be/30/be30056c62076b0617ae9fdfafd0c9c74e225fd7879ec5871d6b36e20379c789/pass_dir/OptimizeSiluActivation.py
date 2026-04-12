import torch
import triton
import triton.language as tl

@triton.jit
def triton_silu_kernel(
    x_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a contiguous block of data
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    if tl.any(mask):
        # Load input
        x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
        
        # Compute SiLU: x * sigmoid(x) = x / (1 + exp(-x))
        # Using Triton operations for computation
        sigmoid_x = 1.0 / (1.0 + tl.exp(-x))
        out = x * sigmoid_x
        
        # Store result
        tl.store(out_ptr + offsets, out, mask=mask)

@torch.fx.wrap
def triton_silu(x):
    """
    Custom SiLU implementation using Triton
    Create output tensor and use Triton kernel for computation
    """
    # Only use allowed operations - create empty tensor
    out = torch.empty_like(x)
    
    n_elements = x.numel()
    BLOCK_SIZE = 1024
    num_programs = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    triton_silu_kernel[(num_programs,)](
        x_ptr=x,
        out_ptr=out,
        n_elements=n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out

def pattern(in_1):
    """Match the SiLU activation operation"""
    tmp_1 = torch.nn.functional.silu(in_1, inplace=True)
    return tmp_1

def replacement_args(in_1):
    """Extract arguments for the replacement"""
    return (in_1,)

def replacement_func():
    """Return the optimized function"""
    return triton_silu