import torch
import triton
import triton.language as tl

def pattern(x):
    """
    Just matches the flatten operation
    Input: [batch, channels, 1, 1] becomes [batch, channels] after flatten(1, -1)
    """
    tmp_2 = x.flatten(1, -1)
    return (tmp_2,)  # Return as tuple to match the model's return structure

def replacement_args(x):
    """Extract arguments for the replacement"""
    return (x,)

@triton.jit
def optimized_flatten_kernel(
    x_ptr, 
    out_ptr, 
    n_elements,
    BLOCK_SIZE: tl.constexpr
):
    """
    Highly optimized flatten operation for [batch, channels, 1, 1] -> [batch, channels]
    Since last two dimensions are 1, this is essentially just a view/reshape operation
    """
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Direct memory copy - no rearrangement needed since flatten(1, -1) on [B,C,1,1] is just a view
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    tl.store(out_ptr + offsets, x, mask=mask)

@torch.fx.wrap
def optimized_flatten(x):
    """
    Highly optimized flatten operation for [batch, channels, 1, 1] -> [batch, channels]
    Since last two dimensions are 1, this is essentially just a view/reshape operation
    """
    input_shape = x.shape
    batch_size, channels = input_shape[0], input_shape[1]
    
    # Output shape: [batch_size, channels] after flatten(1, -1)
    out_shape = [batch_size, channels]
    out = torch.empty(out_shape, dtype=x.dtype, device=x.device)
    
    N = batch_size * channels
    
    # Use smaller block size for better latency
    BLOCK_SIZE = 1024
    num_programs = (N + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    optimized_flatten_kernel[(num_programs,)](
        x_ptr=x,
        out_ptr=out,
        n_elements=N,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out

def replacement_func():
    """Return the optimized function"""
    return optimized_flatten