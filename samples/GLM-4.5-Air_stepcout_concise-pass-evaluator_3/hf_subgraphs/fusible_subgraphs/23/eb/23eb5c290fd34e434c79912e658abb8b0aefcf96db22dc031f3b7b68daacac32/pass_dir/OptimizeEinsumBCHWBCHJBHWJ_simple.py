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

# Triton kernel for optimized einsum operation with fixed C=64
@triton.jit
def einsum_kernel_fixed_c(
    # Input tensors
    query_ptr,  # [B, 64, H, W]
    key_ptr,    # [B, 64, H, 64]
    # Output tensor  
    out_ptr,    # [B, H, W, 64]
    
    # Tensor shapes
    B: tl.constexpr, H: tl.constexpr, W: tl.constexpr,
):
    # Program identifiers for 3D tiling (flatten B and H dimensions)
    bh_idx = tl.program_id(0)   # Combined batch and height: B * H
    w = tl.program_id(1)        # Width dimension
    j = tl.program_id(2)        # J dimension (fixed to 64)
    
    # Extract batch and height from combined index
    h = bh_idx % H
    b = bh_idx // H
    
    # Create masks for boundaries
    if b >= B:
        return
    if h >= H:
        return
    if w >= W:
        return
    if j >= 64:
        return
    
    # Calculate output offset using proper row-major layout
    out_offset = b * H * W * 64 + h * W * 64 + w * 64 + j
    
    # Load the entire C dimension (64 elements) for this position
    c_range = tl.arange(0, 64)
    
    # Calculate flat offset for query tensor [B, 64, H, W]
    query_base_offset = (b * 64 * H * W + h * 64 * W + w * 64)
    query_vals = tl.load(query_ptr + query_base_offset + c_range)
    
    # Calculate flat offset for key tensor [B, 64, H, 64]
    key_base_offset = (b * 64 * H * 64 + h * 64 * 64 + j * 64)
    key_vals = tl.load(key_ptr + key_base_offset + c_range)
    
    # Compute dot product over C dimension
    acc = tl.sum(query_vals * key_vals, axis=0)
    
    # Store the result
    tl.store(out_ptr + out_offset, acc)

# Kernel wrapper
@torch.fx.wrap
def optimized_einsum_fixed_c(query, key):
    """
    Optimized implementation of einsum('bchw,bchj->bhwj') with fixed C=64
    Args:
        query: [B, 64, H, W] tensor
        key: [B, 64, H, 64] tensor  
    Returns:
        out: [B, H, W, 64] tensor
    """
    B, C, H, W = query.shape
    _, _, _, J = key.shape
    
    # Verify fixed dimensions
    assert C == 64, f"Expected C=64, got C={C}"
    assert J == 64, f"Expected J=64, got J={J}"
    
    # Create output tensor
    out = torch.empty((B, H, W, J), dtype=query.dtype, device=query.device)
    
    # Calculate grid dimensions (flatten B and H into first dimension)
    grid = (B * H, W, J)
    
    # Launch kernel
    einsum_kernel_fixed_c[grid](
        query,
        key,
        out,
        B, H, W
    )
    
    return out

# Argument extraction function
def replacement_args(in_2, in_1):
    return (in_2, in_1)

# Replacement function (returns function reference)
def replacement_func():
    return optimized_einsum_fixed_c