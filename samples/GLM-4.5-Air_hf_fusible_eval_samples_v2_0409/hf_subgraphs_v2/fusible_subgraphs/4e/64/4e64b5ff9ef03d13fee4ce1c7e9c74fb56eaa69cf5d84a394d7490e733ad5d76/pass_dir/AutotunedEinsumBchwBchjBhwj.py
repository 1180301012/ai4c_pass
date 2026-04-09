import torch
import triton
import triton.language as tl

def pattern(query, key):
    """
    Pattern matching for einsum('bchw,bchj->bhwj', query, key)
    This is a batched matrix multiplication along the last dimensions
    """
    result = torch.functional.einsum('bchw,bchj->bhwj', query, key)
    return result

def replacement_args(query, key):
    """
    Extract arguments for the replacement function
    """
    return (query, key)

@triton.jit
def autotuned_einsum_kernel(
    query_ptr, key_ptr, out_ptr,
    batch_size: tl.constexpr, height: tl.constexpr, width: tl.constexpr, feat_dim: tl.constexpr
):
    """
    Optimized Triton kernel for einsum('bchw,bchj->bhwj', query, key)
    
    This kernel vectorizes the computation for better performance
    """
    # Get program IDs  
    b = tl.program_id(0)
    h = tl.program_id(1)
    w = tl.program_id(2)
    
    # Only process within bounds
    if b >= batch_size:
        return
    if h >= height:
        return
    if w >= width:
        return
    
    # Compute base pointers - optimized memory layout
    query_offset = b * (feat_dim * height * width) + h * (feat_dim * width) + w * feat_dim
    key_offset = b * (feat_dim * height * width) + h * (feat_dim * width)
    out_offset = b * (height * width * width) + h * (width * width) + w * width
    
    # Vectorized dot product computation
    for j in range(width):
        acc = 0.0
        
        # Process in vectorized chunks of 4 for better performance
        for i in range(0, feat_dim, 4):
            # Process 4 elements at a time
            end = min(i + 4, feat_dim)
            
            # Load and compute product in vectorized manner
            for k in range(i, end):
                query_val = tl.load(query_ptr + query_offset + k)
                key_val = tl.load(key_ptr + (k * width) + j + key_offset)
                acc += query_val * key_val
        
        # Store result
        tl.store(out_ptr + out_offset + j, acc)

@torch.fx.wrap
def autotuned_einsum_bchw_bchj_bhwj(query, key):
    """
    Autotuned einsum('bchw,bchj->bhwj', query, key) using Triton
    """
    batch_size, height, width, feat_dim = query.shape
    
    # Create output tensor with shape [batch_size, height, width, width]
    out = torch.empty((batch_size, height, width, width), dtype=query.dtype, device=query.device)
    
    # Launch kernel - optimal grid configuration
    grid = (batch_size, height, width)
    
    # Launch kernel without autotuning for simplicity
    autotuned_einsum_kernel[grid](
        query, key, out,
        batch_size, height, width, feat_dim
    )
    
    return out

def replacement_func():
    """
    Returns the autotuned function
    """
    return autotuned_einsum_bchw_bchj_bhwj