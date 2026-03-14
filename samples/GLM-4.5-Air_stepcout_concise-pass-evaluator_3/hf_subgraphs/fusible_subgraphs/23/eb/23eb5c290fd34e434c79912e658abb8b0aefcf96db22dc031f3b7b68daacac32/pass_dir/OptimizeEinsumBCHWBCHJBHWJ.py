import torch
import triton
import triton.language as tl

# Pattern matching function - matches the einsum operation
def pattern(in_2, in_1):
    """
    Matches the einsum operation: torch.functional.einsum('bchw,bchj->bhwj', in_2, in_1)
    This performs a batched matrix multiplication where for each (b, h, w) position,
    we compute the dot product between in_2[b,h,w,:] and in_1[b,h,:] across the C dimension.
    """
    result = torch.functional.einsum('bchw,bchj->bhwj', in_2, in_1)
    return result

# Triton kernel for optimized einsum operation
@triton.jit
def einsum_kernel_bchw_bchj_bhwj(
    # Input tensors
    query_ptr,  # [B, C, H, W]
    key_ptr,    # [B, C, H, J]
    # Output tensor  
    out_ptr,    # [B, H, W, J]
    
    # Tensor shapes
    B, C, H, W, J,
    
    # Block sizes  
    BLOCK_SIZE_J: tl.constexpr,  # Block along J dimension
):
    # Program identifiers for 3D tiling (flatten B and H dimensions)
    bh_idx = tl.program_id(0)   # Combined batch and height: B * H
    w = tl.program_id(1)        # Width dimension
    j = tl.program_id(2)        # J dimension
    
    # Extract batch and height from combined index
    h = bh_idx % H
    b = bh_idx // H
    
    # Create masks for boundaries
    b_mask = b < B
    h_mask = h < H
    w_mask = w < W
    if not b_mask:
        return
    if not h_mask:
        return
    if not w_mask:
        return
    
    # Calculate output offsets (without C dimension aggregation)
    out_offset = b * H * W * J + h * W * J + w * J + j
    
    # Load and compute reduction over C dimension
    acc = 0.0
    
    # Load the entire C dimension for this specific (b, h, w, j) position
    # Since C is 64, it should be manageable to load everything at once
    c_range = tl.arange(0, C)
    
    # Calculate flat offset for batch, height, width position
    base_query_offset = b * C * H * W + h * W + w
    base_key_offset = b * C * H * J + h * J + j
    
    # Load query and key vectors for the full C dimension
    query_vals = tl.load(query_ptr + base_query_offset * C + c_range, mask=c_range < C, other=0.0)
    key_vals = tl.load(key_ptr + base_key_offset * C + c_range, mask=c_range < C, other=0.0)
    
    # Compute dot product over C dimension
    acc = tl.sum(query_vals * key_vals, axis=0)
    
    # Store the result
    tl.store(out_ptr + out_offset, acc)

# Kernel wrapper
@torch.fx.wrap
def optimized_einsum_bchw_bchj_bhwj(query, key):
    """
    Optimized implementation of einsum('bchw,bchj->bhwj')
    Args:
        query: [B, C, H, W] tensor
        key: [B, C, H, J] tensor  
    Returns:
        out: [B, H, W, J] tensor
    """
    B, C, H, W = query.shape
    _, _, _, J = key.shape
    
    # Create output tensor
    out = torch.empty((B, H, W, J), dtype=query.dtype, device=query.device)
    
    # Choose block sizes based on tensor dimensions
    BLOCK_SIZE_J = 8
    
    # Calculate grid dimensions (flatten B and H into first dimension)
    grid = (B * H, W, J)
    
    # Launch kernel
    einsum_kernel_bchw_bchj_bhwj[grid](
        query,
        key,
        out,
        B, C, H, W, J,
        BLOCK_SIZE_J
    )
    
    return out

# Argument extraction function
def replacement_args(in_2, in_1):
    return (in_2, in_1)

# Replacement function (returns function reference)
def replacement_func():
    return optimized_einsum_bchw_bchj_bhwj