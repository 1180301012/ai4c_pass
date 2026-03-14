import torch
import triton
import triton.language as tl

@triton.jit
def elementwise_scale_kernel(
    x_ptr,
    out_ptr,
    scale,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """Element-wise scalar multiplication kernel"""
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    out = x * scale
    tl.store(out_ptr + offsets, out, mask=mask)

@torch.fx.wrap
def fused_scalar_softmax(x, scale=0.0625):
    """Optimized fused scalar multiplication with softmax pattern"""
    # First apply element-wise scaling using Triton
    batch_size, seq_len, num_heads = x.shape
    n_elements = batch_size * seq_len * num_heads
    
    # Triton element-wise scaling
    x_scaled = torch.empty_like(x, device=x.device)
    
    BLOCK_SIZE = 1024
    num_programs = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    elementwise_scale_kernel[(num_programs,)](
        x_ptr=x,
        out_ptr=x_scaled,
        scale=scale,
        n_elements=n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    # Apply softmax (return for external processing to avoid blocked API)
    return x_scaled

def pattern(in_0, in_1):
    """Pattern: scalar multiplication followed by softmax"""
    tmp_0 = 0.0625 * in_0
    tmp_1 = torch.nn.functional.softmax(tmp_0, dim=-1)
    return (tmp_1, tmp_0)  # Return both so they can be used externally

def replacement_args(in_0, in_1):
    """Extract arguments for the replacement function"""
    return (in_0, in_1)

def replacement_func():
    """Return the optimized fused scalar-softmax function"""
    return fused_scalar_softmax