import torch
import triton
import triton.language as tl

def pattern(x, y):
    tmp_8 = x.unsqueeze(1)
    tmp_9 = tmp_8.unsqueeze(0)
    result = y + tmp_9
    return result

def replacement_args(x, y):
    return (x, y)

@triton.jit
def bias_addition_kernel(
    attention_scores_ptr,
    attn_mask_ptr,
    output_ptr,
    batch_size,
    num_heads,
    channels,
    height,
    width,
    BLOCK_SIZE: tl.constexpr,
):
    # Calculate total elements per head dimension batch
    total_elements = num_heads * channels * height * width
    pid = tl.program_id(0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offset < total_elements
    
    # Compute indices
    linear_idx = offset
    head_idx = linear_idx // (channels * height * width)
    remaining = linear_idx % (channels * height * width)
    channel_idx = remaining // (height * width)
    remaining = remaining % (height * width)
    y_idx = remaining // width
    x_idx = remaining % width
    
    # Load attention scores (already expanded to [1, num_heads, channels, height, width])
    attention_scores = tl.load(attention_scores_ptr + offset, mask=mask, other=0.0)
    
    # Load attention mask (expanded from [batch_size, height, width] to [batch_size, num_heads, height, width])
    mask_offset = pid * (height * width) + (y_idx * width + x_idx)
    mask_value = tl.load(attn_mask_ptr + mask_offset, mask=mask, other=0.0)
    
    # Add bias
    out = attention_scores + mask_value
    
    # Store result
    tl.store(output_ptr + offset, out, mask=mask)

@torch.fx.wrap
def optimized_bias_addition(attention_scores, attn_mask):
    """
    Optimized fusion of bias mask expansion and addition
    Handles: unsqueeze(1) + unsqueeze(0) + addition
    """
    # Input shapes: attention_scores [1, num_heads, channels, height, width]
    #              attn_mask [batch_size, height, width]
    
    batch_size = attention_scores.shape[0]
    num_heads = attention_scores.shape[1]
    channels = attention_scores.shape[2]
    height = attention_scores.shape[3]
    width = attention_scores.shape[4]
    
    total_elements = batch_size * num_heads * channels * height * width
    output = torch.empty_like(attention_scores)
    
    BLOCK_SIZE = 1024
    num_programs = (total_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Expand attn_mask and add in one kernel
    bias_addition_kernel[(num_programs,)](
        attention_scores_ptr=attention_scores,
        attn_mask_ptr=attn_mask,
        output_ptr=output,
        batch_size=batch_size,
        num_heads=num_heads,
        channels=channels,
        height=height,
        width=width,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return output

def replacement_func():
    return optimized_bias_addition