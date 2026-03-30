import torch
import triton
import triton.language as tl

def pattern(in_0):
    """Match SiLU operation with explicit parameters"""
    tmp_0 = torch.nn.functional.silu(in_0, inplace=False)
    return (tmp_0,)

def replacement_args(in_0):
    """Extract arguments for the fused kernel"""
    return (in_0,)

@triton.jit
def fused_silu_mul_kernel(
    x_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """Optimized SiLU kernel
    SiLU(x) = x * sigmoid(x)
    """
    # Each program handles a contiguous block of data
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load input
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    
    # Compute SiLU operation: x * sigmoid(x)
    sigmoid_x = tl.sigmoid(x)
    out = x * sigmoid_x
    
    # Store result
    tl.store(out_ptr + offsets, out, mask=mask)

@torch.fx.wrap
def optimized_silu(in_0):
    """Wrapper function for optimized SiLU operation"""
    # Calculate total number of elements
    n_elements = in_0.numel()
    
    # Optimized block size for this tensor size (1, 257, 1024)
    # 263,168 total elements - 1024 gives good GPU occupancy
    BLOCK_SIZE = 1024
    num_programs = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Create output tensor with same dtype and device as inputs
    out = torch.empty_like(in_0)
    
    # Launch the kernel
    fused_silu_mul_kernel[(num_programs,)](
        in_0,
        out,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out

def replacement_func():
    """Return the optimized kernel function"""
    return optimized_silu