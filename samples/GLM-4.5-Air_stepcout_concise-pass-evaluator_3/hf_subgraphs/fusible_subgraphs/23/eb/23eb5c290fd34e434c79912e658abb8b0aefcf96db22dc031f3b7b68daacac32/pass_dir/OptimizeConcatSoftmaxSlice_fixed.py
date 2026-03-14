import torch
import triton
import triton.language as tl

# Pattern matching function - matches the concat, softmax, and slice operations
def pattern(tmp_0, tmp_1):
    """
    Matches the sequence:
    tmp_2 = torch.cat([tmp_0, tmp_1], dim=-1)
    tmp_3 = torch.nn.functional.softmax(tmp_2, dim=-1)  
    tmp_4 = tmp_3[Ellipsis, slice(None, 64, None)]
    return (tmp_3, tmp_4)
    """
    tmp_2 = torch.cat([tmp_0, tmp_1], dim=-1)
    tmp_3 = torch.nn.functional.softmax(tmp_2, dim=-1)
    tmp_4 = tmp_3[Ellipsis, slice(None, 64, None)]
    return tmp_3, tmp_4

# Triton kernel for fused concat-softmax operation
@triton.jit
def concat_softmax_kernel(
    # Input tensors
    energy_H_1_ptr,    # [B, H, W, K1] energy tensor (first 64 elements)
    einsum_out_ptr,     # [B, H, W, 64] einsum result tensor (next 64 elements)
    # Output tensors
    softmax_out_ptr,    # [B, H, W, K1+64] full softmax output
    slice_out_ptr,      # [B, H, W, 64] sliced softmax output
    
    # Tensor shapes
    B, H, W, K1,
    SLICE_SIZE: tl.constexpr,  # Number of elements to slice (64)
    
    # Block sizes
    BLOCK_SIZE_K: tl.constexpr,
):
    # Program identifiers for 3D tiling (flatten B and H dimensions)
    bh_idx = tl.program_id(0)   # Combined batch and height: B * H
    w = tl.program_id(1)        # Width dimension
    k = tl.program_id(2)        # K dimension (K1+64)
    
    # Extract batch and height from combined index
    h = bh_idx % H
    b = bh_idx // H
    
    # Create boundary masks
    b_mask = b < B
    h_mask = h < H
    w_mask = w < W
    
    if not (b_mask and h_mask and w_mask):
        return
    
    # Calculate global offsets
    energy_offset = b * H * W * K1 + h * W * K1 + w
    einsum_offset = b * H * W * SLICE_SIZE + h * W * SLICE_SIZE
    softmax_offset = b * H * W * (K1 + SLICE_SIZE) + h * W * (K1 + SLICE_SIZE) + w
    
    # Compute softmax over concatenated tensor (K1 + 64 elements)
    k_total = K1 + SLICE_SIZE
    
    # Find max and compute softmax for the current (b, h, w, k) position
    if k < k_total:
        # Calculate current K position in the concatenated space
        if k < K1:
            # First part: energy_H_1
            local_k = k
            current_val = tl.load(energy_H_1_ptr + energy_offset + local_k)
        else:
            # Second part: einsum result
            local_k = k - K1
            current_val = tl.load(einsum_out_ptr + einsum_offset + local_k)
        
        # Compute max over the entire K dimension for this (b, h, w)
        max_val = -tl.inf
        
        # Find max in energy part
        for k_pos in range(0, K1, BLOCK_SIZE_K):
            k_remaining = K1 - k_pos
            k_block = min(k_remaining, BLOCK_SIZE_K)
            k_mask = tl.arange(0, k_block) < k_remaining
            
            energy_part = tl.load(energy_H_1_ptr + energy_offset + k_pos, mask=k_mask, other=-tl.inf)
            block_max = tl.max(energy_part)
            max_val = tl.maximum(max_val, block_max)
        
        # Find max in einsum part
        for k_pos in range(0, SLICE_SIZE, BLOCK_SIZE_K):
            k_remaining = SLICE_SIZE - k_pos
            k_block = min(k_remaining, BLOCK_SIZE_K)
            k_mask = tl.arange(0, k_block) < k_remaining
            
            einsum_part = tl.load(einsum_out_ptr + einsum_offset + k_pos, mask=k_mask, other=-tl.inf)
            block_max = tl.max(einsum_part)
            max_val = tl.maximum(max_val, block_max)
        
        # Compute softmax value
        exp_val = tl.exp(current_val - max_val)
        out_val = exp_val
        
        # Store results
        softmax_full_offset = softmax_offset + k
        tl.store(softmax_out_ptr + softmax_full_offset, out_val)
        
        # Also store the sliced part (first 64 elements)
        if k < SLICE_SIZE:
            slice_local_offset = b * H * W * SLICE_SIZE + h * W * SLICE_SIZE + w * SLICE_SIZE + k
            tl.store(slice_out_ptr + slice_local_offset, out_val)

# Optimized kernel wrapper
@torch.fx.wrap
def optimized_concat_softmax_slice(energy_H_1, einsum_result):
    """
    Optimized implementation of concat + softmax + slice operations
    Args:
        energy_H_1: [B, H, W, K] energy tensor (includes the concatenate dimension)
        einsum_result: [B, H, W, 64] einsum result tensor
    Returns:
        softmax_out: [B, H, W, K+64] full softmax output (note: K includes original concatenate dim)
        slice_out: [B, H, W, 64] sliced softmax output
    """
    B, H, W, K1 = energy_H_1.shape
    SLICE_SIZE = 64  # Fixed slice size from the pattern
    
    # Create output tensors
    # The concatenated tensor has shape [B, H, W, K1 + SLICE_SIZE]
    softmax_out = torch.empty((B, H, W, K1 + SLICE_SIZE), dtype=energy_H_1.dtype, device=energy_H_1.device)
    slice_out = torch.empty((B, H, W, SLICE_SIZE), dtype=energy_H_1.dtype, device=energy_H_1.device)
    
    # Set block sizes
    BLOCK_SIZE_K = 32
    
    # Calculate grid dimensions (flatten B and H into first dimension)
    grid = (B * H, W, K1 + SLICE_SIZE)
    
    # Launch kernel
    concat_softmax_kernel[grid](
        energy_H_1,
        einsum_result,
        softmax_out,
        slice_out,
        B, H, W, K1, SLICE_SIZE,
        BLOCK_SIZE_K
    )
    
    return softmax_out, slice_out

# Argument extraction function
def replacement_args(tmp_0, tmp_1):
    return (tmp_0, tmp_1)

# Replacement function (returns function reference)
def replacement_func():
    return optimized_concat_softmax_slice