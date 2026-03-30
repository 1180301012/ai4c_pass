import torch
import triton
import triton.language as tl

def pattern(in_9, tmp_9):
    """
    Pattern matches two sigmoid operations:
    - tmp_10 = in_9.sigmoid()
    - tmp_11 = tmp_9.sigmoid()
    """
    tmp_10 = in_9.sigmoid()
    tmp_11 = tmp_9.sigmoid()
    return tmp_10, tmp_11

def replacement_args(in_9, tmp_9):
    return (in_9, tmp_9)

@triton.jit
def fused_sigmoid_kernel(
    in_9_ptr, tmp_9_ptr,
    tmp_10_ptr, tmp_11_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fused kernel that computes two sigmoid operations in parallel
    """
    # Each program handles a contiguous block of data
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load inputs
    in_9_val = tl.load(in_9_ptr + offsets, mask=mask, other=0.0)
    tmp_9_val = tl.load(tmp_9_ptr + offsets, mask=mask, other=0.0)
    
    # Compute both sigmoid operations in parallel
    tmp_10_val = 1.0 / (1.0 + tl.exp(-in_9_val))
    tmp_11_val = 1.0 / (1.0 + tl.exp(-tmp_9_val))
    
    # Store results
    tl.store(tmp_10_ptr + offsets, tmp_10_val, mask=mask)
    tl.store(tmp_11_ptr + offsets, tmp_11_val, mask=mask)

@torch.fx.wrap
def fused_sigmoid_computation(in_9, tmp_9):
    """
    Wrapper function for fused sigmoid computation
    """
    # Determine tensor size - handle both [300, 1, 256] and [300, 256] shapes
    if len(in_9.shape) == 3:
        n_elements = in_9.shape[0] * in_9.shape[1] * in_9.shape[2]
    else:
        n_elements = in_9.shape[0] * in_9.shape[1]
    
    # Create output tensors
    tmp_10 = torch.empty_like(in_9)
    tmp_11 = torch.empty_like(tmp_9)
    
    # Launch kernel
    BLOCK_SIZE = 1024
    grid_size = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    fused_sigmoid_kernel[grid_size](
        in_9_ptr=in_9,
        tmp_9_ptr=tmp_9,
        tmp_10_ptr=tmp_10,
        tmp_11_ptr=tmp_11,
        n_elements=n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return tmp_10, tmp_11

def replacement_func():
    return fused_sigmoid_computation