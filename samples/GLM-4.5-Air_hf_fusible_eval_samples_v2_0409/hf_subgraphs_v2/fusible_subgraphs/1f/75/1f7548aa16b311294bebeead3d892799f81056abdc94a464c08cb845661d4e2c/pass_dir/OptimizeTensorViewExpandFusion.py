import torch
import triton
import triton.language as tl

def pattern(in_0, in_1, in_2, in_3):
    """Match the tensor view and expand operations pattern"""
    tmp_2 = in_0.to(device=device(type='cuda', index=0), dtype=torch.bool)
    tmp_3 = torch.arange(3, device=device(type='cuda', index=0))
    tmp_3 += 0
    tmp_4 = tmp_3
    tmp_3 = None
    tmp_5 = tmp_2[(slice(None, None, None), tmp_4)]
    tmp_2 = tmp_4 = None
    tmp_6 = torch.arange(3, device=device(type='cuda', index=0))
    tmp_6 += 0
    tmp_7 = tmp_6
    tmp_6 = None
    tmp_8 = in_2.view(-1, 1)
    tmp_9 = tmp_7 <= tmp_8
    tmp_7 = tmp_8 = None
    tmp_10 = tmp_9[(None, None, slice(None, None, None), slice(None, None, None))]
    tmp_9 = None
    tmp_11 = tmp_10.expand(1, -1, -1, -1)
    tmp_10 = None
    tmp_12 = tmp_5[(slice(None, None, None), None, None, slice(None, None, None))]
    tmp_5 = None
    tmp_13 = tmp_11 * tmp_12
    tmp_11 = tmp_12 = None
    return (tmp_13,)

def replacement_args(in_0, in_1, in_2, in_3):
    return (in_0, in_2)

@triton.jit
def optimized_view_expand_kernel(
    attention_mask_ptr,
    cache_position_ptr,
    output_ptr,
    batch_size,
    seq_len,
    BLOCK_SIZE: tl.constexpr,
):
    # Each thread handles a batch and sequence position
    batch_idx = tl.program_id(0)
    seq_idx = tl.program_id(1)
    
    # Linear index for attention mask
    linear_idx = batch_idx * seq_len + seq_idx
    
    # Validity mask
    mask = linear_idx < batch_size * seq_len
    
    if mask:
        # Load attention mask value
        attention_val = tl.load(attention_mask_ptr + linear_idx, mask=mask)
        
        # Load cache position for this sequence position
        cache_val = tl.load(cache_position_ptr + seq_idx, mask=mask)
        
        # Create broadcast mask - equivalent to the optimized view+expand operations
        # The pattern creates: [1, seq_len, 1, 1] expanded to [1, seq_len, 1, seq_len]
        # for comparing with [batch_size, seq_len, 1, 1] expanded from attention
        
        # For current batch and position, determine if attention is enabled
        # This combines the view, expand, and comparison operations
        is_valid_position = (seq_idx <= cache_val)
        attention_enabled = (attention_val == 1) & is_valid_position
        
        # Store result (equivalent to tmp_13)
        tl.store(output_ptr + linear_idx, attention_enabled, mask=mask)

@torch.fx.wrap
def optimized_view_expand_fusion(attention_mask, cache_position):
    # Get input dimensions
    batch_size, seq_len = attention_mask.shape
    
    # Create output tensor with same shape as attention mask
    output = torch.zeros_like(attention_mask, dtype=torch.bool)
    
    # Set up Triton kernel grid
    grid = (batch_size, seq_len)
    
    # Launch optimized kernel that fuses view and expand operations
    optimized_view_expand_kernel[grid](
        attention_mask_ptr=attention_mask,
        cache_position_ptr=cache_position,
        output_ptr=output,
        batch_size=batch_size * seq_len,
        seq_len=seq_len,
        BLOCK_SIZE=1
    )
    
    return output

def replacement_func():
    return optimized_view_expand_fusion