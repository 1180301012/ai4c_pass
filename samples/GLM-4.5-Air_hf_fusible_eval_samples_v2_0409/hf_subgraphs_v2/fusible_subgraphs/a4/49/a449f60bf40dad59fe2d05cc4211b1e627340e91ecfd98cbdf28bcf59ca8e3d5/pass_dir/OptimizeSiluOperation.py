import torch
import triton
import triton.language as tl

@triton.jit
def silu_kernel_optimized(
    x_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """Optimized SiLU kernel: x * sigmoid(x)"""
    # Each program handles a contiguous block of data
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load input data
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    
    # Compute SiLU: x * sigmoid(x)
    # For numerical stability, use stable sigmoid: 1 / (1 + exp(-x))
    sigmoid_x = 1.0 / (1.0 + tl.exp(-x))
    out = x * sigmoid_x
    
    # Store output
    tl.store(out_ptr + offsets, out, mask=mask)

def pattern(in_0, in_1, in_2):
    """Match the complete data flow from the original computation"""
    # Apply SiLU to in_0 
    tmp_0 = torch.nn.functional.silu(in_0, inplace=True)
    
    # Create detached versions - these are the same values but ensure we match the original behavior
    tmp_1 = in_1.detach()
    tmp_2 = in_2.detach()
    tmp_3 = tmp_0.detach()
    
    # Return exactly what the original function returns
    return (tmp_1, tmp_2, tmp_3, tmp_0)

def replacement_args(in_0, in_1, in_2):
    """Extract arguments for the replacement"""
    return (in_0, in_1, in_2)

@torch.fx.wrap
def optimized_forward_computation(in_0, in_1, in_2):
    """Optimized forward computation for the entire graph"""
    # Apply optimized SiLU to in_0
    if in_0.numel() == 0:
        tmp_0 = in_0
    else:
        N = in_0.numel()
        BLOCK_SIZE = 1024
        num_programs = (N + BLOCK_SIZE - 1) // BLOCK_SIZE
        
        # Create output tensor with same properties as input
        tmp_0 = torch.empty_like(in_0)
        
        # Launch optimized Triton kernel
        silu_kernel_optimized[(num_programs,)](
            x_ptr=in_0,
            out_ptr=tmp_0,
            n_elements=N,
            BLOCK_SIZE=BLOCK_SIZE,
        )
    
    # Use detach operations (these maintain the original behavior)
    tmp_1 = in_1.detach()
    tmp_2 = in_2.detach()
    tmp_3 = tmp_0.detach()
    
    # Return exactly what the original function returns
    return (tmp_1, tmp_2, tmp_3, tmp_0)

def replacement_func():
    """Return the optimized function"""
    return optimized_forward_computation