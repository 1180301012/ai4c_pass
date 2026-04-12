import torch
import triton
import triton.language as tl

@triton.jit
def triton_expand_kernel(
    x_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a contiguous block of data
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load input - need to handle the indexing to expand from [2, 128] to [1, 1, 2, 128]
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    
    # Store at the correct position in the expanded tensor
    tl.store(out_ptr + offsets, x, mask=mask)

@torch.fx.wrap
def triton_expand_simple(x):
    """
    Simple tensor expansion using only allowed operations
    Create expanded tensor and use Triton kernel to copy data
    """
    expanded_shape = (1, 1) + x.shape  # [1, 1, 2, 128]
    out = torch.empty(expanded_shape, dtype=x.dtype, device=x.device)
    
    # Since we're expanding with None dimensions, we just copy the original data
    # to the appropriate positions in the expanded tensor
    # For [1, 1, 2, 128] from [2, 128], we copy to out[0, 0]
    n_elements = x.numel()
    BLOCK_SIZE = 1024
    num_programs = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    triton_expand_kernel[(num_programs,)](
        x_ptr=x,
        out_ptr=out[0, 0],  # Target the slice where data should go
        n_elements=n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out

def pattern(in_0):
    """Match the tensor expansion operation"""
    tmp_7 = in_0[(None, None, slice(None, None, None))]
    return tmp_7

def replacement_args(in_0):
    """Extract arguments for the replacement"""
    return (in_0,)

def replacement_func():
    """Return the optimized function"""
    return triton_expand_simple