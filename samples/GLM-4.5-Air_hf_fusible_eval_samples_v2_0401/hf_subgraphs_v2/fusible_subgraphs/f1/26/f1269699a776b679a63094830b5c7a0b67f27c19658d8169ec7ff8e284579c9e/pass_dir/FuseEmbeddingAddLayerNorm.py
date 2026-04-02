import torch
import triton
import triton.language as tl

def pattern(seq_len):
    """Pattern: attention mask computation"""
    tmp_10 = torch.arange(seq_len, dtype=torch.int64)
    tmp_11 = tmp_10[(slice(None, None, None), None)]
    tmp_12 = torch.arange(seq_len, dtype=torch.int64)
    tmp_13 = tmp_12[(None, slice(None, None, None))]
    tmp_14 = tmp_13 - tmp_11
    tmp_15 = -tmp_14
    tmp_16 = tmp_15 < 0
    tmp_17 = tmp_16.to(torch.int64)
    tmp_18 = tmp_17 * 16
    tmp_19 = 0 + tmp_18
    tmp_20 = torch.abs(tmp_15)
    tmp_21 = tmp_20 < 8
    tmp_22 = tmp_20.float()
    tmp_23 = tmp_22 / 8
    tmp_24 = torch.log(tmp_23)
    tmp_25 = tmp_24 / 2.772588722239781
    tmp_26 = tmp_25 * 8
    tmp_27 = tmp_26.to(torch.int64)
    tmp_28 = 8 + tmp_27
    tmp_29 = torch.full_like(tmp_28, 15)
    tmp_30 = torch.min(tmp_28, tmp_29)
    tmp_31 = torch.where(tmp_21, tmp_20, tmp_30)
    tmp_19 += tmp_31
    return tmp_19

def replacement_args(seq_len):
    return (seq_len,)

@triton.jit
def optimized_attention_mask_kernel(
    mask_ptr,
    seq_len,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    """Optimized kernel for attention mask computation"""
    row = tl.program_id(0)
    col = tl.program_id(1)
    
    if row >= seq_len or col >= seq_len:
        return
    
    # Compute difference (col - row)
    diff = col - row
    abs_diff = tl.abs(diff)
    
    # Base penalty for negative differences (causal masking)
    if diff < 0:
        base_penalty = 16
        
        # Compute attention pattern based on distance
        if abs_diff < 8:
            # Linear interpolation for close tokens
            # Avoid log(0) by adding small epsilon
            float_val = (abs_diff + 1e-6) / 8.0
            log_val = tl.log(float_val) 
            scaled_val = log_val / 2.772588722239781  # ln(8)
            rounded_val = tl.round(scaled_val * 8.0)
            clipped_val = tl.maximum(8.0, rounded_val)
            pattern_val = tl.minimum(clipped_val, 15.0)
            mask_val = pattern_val
        else:
            # Use maximum penalty for distant tokens
            mask_val = 15.0
        
        # Combine base penalty with attention pattern
        final_penalty = base_penalty + mask_val
    else:
        # Zero penalty for future tokens in causal mask
        final_penalty = 0
    
    # Store result
    offset = row * seq_len + col
    tl.store(mask_ptr + offset, final_penalty)

@torch.fx.wrap
def compute_attention_mask_optimized(seq_len):
    """Wrapper for optimized attention mask computation"""
    mask = torch.empty((seq_len, seq_len), dtype=torch.int64, device='cuda:0')
    
    # Block sizes for better memory coalescing
    BLOCK_SIZE_M = 32
    BLOCK_SIZE_N = 32
    
    # Grid size based on sequence length with blocks
    grid_m = (seq_len + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M
    grid_n = (seq_len + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N
    grid = (grid_m, grid_n)
    
    optimized_attention_mask_kernel[grid](
        mask,
        seq_len,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
    )
    
    return mask

def replacement_func():
    return compute_attention_mask_optimized