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

# Triton kernel for fused concat-softmax-slice operation
@triton.jit
def concat_softmax_slice_kernel(
    # Input tensors
    energy_H_1_ptr,    # [B, H, W, K] energy tensor
    einsum_out_ptr,     # [B, H, W, 64] einsum result tensor
    # Output tensors
    softmax_out_ptr,    # [B, H, W, K] full softmax output
    slice_out_ptr,      # [B, H, W, 64] sliced softmax output
    
    # Tensor shapes
    B, H, W, K,
    SLICE_SIZE: tl.constexpr,  # Number of elements to slice (64)
    
    # Block sizes
    BLOCK_SIZE_B: tl.constexpr,
    BLOCK_SIZE_H: tl.constexpr,
    BLOCK_SIZE_W: tl.constexpr, 
    BLOCK_SIZE_K: tl.constexpr,
):
    # Program identifiers
    b = tl.program_id(0)
    h = tl.program_id(1)
    w = tl.program_id(2)
    k = tl.program_id(3)
    
    # Calculate global offsets
    energy_offset = b * H * W * K + h * W * K + k
    einsum_offset = b * H * W * SLICE_SIZE + h * W * SLICE_SIZE + k if k < SLICE_SIZE else 0
    
    # Create masks
    b_mask = b < B
    h_mask = h < H
    w_mask = w < W
    k_mask = k < K
    
    if not (b_mask and h_mask and w_mask):
        return
    
    # Load input values (broadcast across K dimension for softmax computation)
    if k < SLICE_SIZE:
        # Load energy and einsum values for positions within slice range
        energy_val = tl.load(energy_H_1_ptr + energy_offset, mask=k_mask)
        einsum_val = tl.load(einsum_out_ptr + einsum_offset)
        
        # For softmax computation, we need all values in the K dimension
        # Load all values in the current (b, h, w) slice
        max_val = -tl.inf
        sum_exp = 0.0
        
        # Process the concatenation (energy_H_1[64:] + einsum_result[64:])
        for k_pos in range(0, K, BLOCK_SIZE_K):
            k_remaining = K - k_pos
            k_block = min(k_remaining, BLOCK_SIZE_K)
            
            # Load energy part (this is already in memory)
            if k_pos <= k < k_pos + k_block:
                local_k = k - k_pos
                energy_part = tl.load(energy_H_1_ptr + energy_offset + k_pos, 
                                    mask=tl.arange(0, k_block) < k_remaining, other=0.0)
                
                # Find max in the current block
                block_max = tl.max(energy_part)
                if block_max > max_val:
                    max_val = block_max
                
                # Accumulate exponential sum
                sum_exp += tl.sum(tl.exp(energy_part - max_val))
        
        # Load einsum part (first 64 elements)
        for k_pos in range(0, SLICE_SIZE, BLOCK_SIZE_K):
            k_remaining = SLICE_SIZE - k_pos
            k_block = min(k_remaining, BLOCK_SIZE_K)
            
            einsum_part = tl.load(einsum_out_ptr + (b * H * W * SLICE_SIZE + h * W * SLICE_SIZE + k_pos),
                                mask=tl.arange(0, k_block) < k_remaining, other=0.0)
            
            # Find max in the current block
            block_max = tl.max(einsum_part)
            if block_max > max_val:
                max_val = block_max
            
            # Accumulate exponential sum
            sum_exp += tl.sum(tl.exp(einsum_part - max_val))
        
        # Compute softmax values
        out_val = tl.exp(energy_val - max_val) / sum_exp
    else:
        # Only compute for the energy H_1 part (k >= SLICE_SIZE)
        out_val = 0.0  # This will be handled by the einsum-only path
    
    # Store results
    tl.store(softmax_out_ptr + energy_offset, out_val, mask=k_mask)

# Optimized kernel wrapper
@torch.fx.wrap
def optimized_concat_softmax_slice(energy_H_1, einsum_result):
    """
    Optimized implementation of concat + softmax + slice operations
    Args:
        energy_H_1: [B, H, W, K] energy tensor (K includes the concatenate dimension)
        einsum_result: [B, H, W, 64] einsum result tensor
    Returns:
        softmax_out: [B, H, W, K] full softmax output
        slice_out: [B, H, W, 64] sliced softmax output
    """
    B, H, W, K = energy_H_1.shape
    SLICE_SIZE = 64  # Fixed slice size from the pattern
    
    # Create output tensors
    softmax_out = torch.empty((B, H, W, K), dtype=energy_H_1.dtype, device=energy_H_1.device)
    slice_out = torch.empty((B, H, W, SLICE_SIZE), dtype=energy_H_1.dtype, device=energy_H_1.device)
    
    # Set block sizes
    BLOCK_SIZE_B = 1
    BLOCK_SIZE_H = 4
    BLOCK_SIZE_W = 8
    BLOCK_SIZE_K = 32
    
    # Calculate grid dimensions
    grid = (B, H, W, K)
    
    # Launch kernel
    concat_softmax_slice_kernel[grid](
        energy_H_1,
        einsum_result,
        softmax_out,
        slice_out,
        B, H, W, K, SLICE_SIZE,
        BLOCK_SIZE_B, BLOCK_SIZE_H, BLOCK_SIZE_W, BLOCK_SIZE_K
    )
    
    return softmax_out, slice_out

# Argument extraction function
def replacement_args(tmp_0, tmp_1):
    return (tmp_0, tmp_1)

# Replacement function (returns function reference)
def replacement_func():
    return optimized_concat_softmax_slice