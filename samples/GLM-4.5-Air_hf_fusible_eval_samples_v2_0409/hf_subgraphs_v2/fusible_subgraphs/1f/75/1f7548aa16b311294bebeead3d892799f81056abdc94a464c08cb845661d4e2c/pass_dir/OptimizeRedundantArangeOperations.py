import torch
import triton
import triton.language as tl

def pattern(in_0, in_1, in_2, in_3):
    """Match the redundant arange operations pattern"""
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
    return (tmp_13,)

def replacement_args(in_0, in_1, in_2, in_3):
    return (in_0, in_2)

@triton.jit
def optimized_arange_kernel(
    attention_mask_ptr,
    cache_position_ptr,
    output_ptr,
    batch_size,
    seq_len,
    attention_mask_shape0,
    BLOCK_SIZE: tl.constexpr,
):
    # Calculate grid positions
    batch_idx = tl.program_id(0)
    seq_idx = tl.program_id(1)
    
    # Create indices for this position
    global_idx = batch_idx * seq_len + seq_idx
    
    # Only process valid positions
    mask = (global_idx < batch_size * seq_len) & (batch_idx < attention_mask_shape0)
    
    if mask:
        # Load attention mask value (0 or 1 for attention)
        attention_val = tl.load(attention_mask_ptr + global_idx, mask=mask)
        
        # Create cache position index (equivalent to arange)
        cache_pos = seq_idx
        
        # Load cache position value
        cache_val = tl.load(cache_position_ptr + seq_idx, mask=mask)
        
        # Create comparison tensor (equivalent to arange <= cache_position)
        # For position seq_idx, the condition is seq_idx <= cache_val
        attention_enabled = (attention_val == 1) & (seq_idx <= cache_val)
        
        # Store the optimized result
        tl.store(output_ptr + global_idx, attention_enabled, mask=mask)

@torch.fx.wrap
def optimized_arange_computation(attention_mask, cache_position):
    # Get input dimensions
    batch_size, seq_len = attention_mask.shape
    attention_mask_shape0 = attention_mask.shape[0]
    
    # Create output tensor
    output = torch.zeros_like(attention_mask, dtype=torch.bool)
    
    # Set up kernel launch configuration
    grid = (batch_size, seq_len)
    
    # Launch the optimized kernel
    optimized_arange_kernel[grid](
        attention_mask_ptr=attention_mask,
        cache_position_ptr=cache_position,
        output_ptr=output,
        batch_size=batch_size * seq_len,
        seq_len=seq_len,
        attention_mask_shape0=attention_mask_shape0,
        BLOCK_SIZE=1
    )
    
    return output

def replacement_func():
    return optimized_arange_computation