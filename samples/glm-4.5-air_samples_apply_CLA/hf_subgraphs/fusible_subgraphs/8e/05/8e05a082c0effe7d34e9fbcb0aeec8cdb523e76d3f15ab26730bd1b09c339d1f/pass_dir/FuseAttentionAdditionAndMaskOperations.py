import torch
import triton
import triton.language as tl

def pattern(in_1, tmp_5, in_2):
    tmp_6 = in_1 + tmp_5
    # Reshape to add batch dimension: [1, heads, features, height, width]
    tmp_7 = tmp_6.unsqueeze(0).view(1, in_1.size(0), in_1.size(1), in_1.size(2), in_1.size(3))
    tmp_8 = in_2.unsqueeze(1)
    tmp_9 = tmp_8.unsqueeze(0)
    tmp_10 = tmp_7 + tmp_9
    tmp_11 = tmp_10.view(-1, in_1.size(1), in_1.size(2), in_1.size(3))
    return tmp_11

def replacement_args(in_1, tmp_5, in_2):
    return (in_1, tmp_5, in_2)

@triton.jit
def fused_attention_kernel(
    attention_scores_ptr,
    bias_scores_ptr,
    attn_mask_ptr,
    out_ptr,
    batch_size,
    num_heads,
    features,
    height,
    width,
    total_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < total_elements
    
    # Calculate element coordinates within attention scores [num_heads, features, height, width]
    linear_idx = offsets
    head_idx = linear_idx // (features * height * width)
    rem_idx = linear_idx % (features * height * width)
    feature_idx = rem_idx // (height * width)
    rem_idx = rem_idx % (height * width)
    h_idx = rem_idx // width
    w_idx = rem_idx % width
    
    # Load attention scores [num_heads, features, height, width]
    attention_scores = tl.load(attention_scores_ptr + linear_idx, mask=mask, other=0.0)
    
    # Load bias scores [1, features, height, width] - broadcast across heads
    bias_scores = tl.load(bias_scores_ptr + feature_idx * height * width + h_idx * width + w_idx, mask=mask, other=0.0)
    
    # Load attention mask [num_heads, height, width] - broadcast across features
    mask_linear_idx = head_idx * (height * width) + h_idx * width + w_idx
    attn_mask = tl.load(attn_mask_ptr + mask_linear_idx, mask=mask, other=0.0)
    
    # Apply broadcasting: bias is [1, features, height, width] -> [num_heads, features, height, width]
    # mask is [num_heads, height, width] -> [num_heads, features, height, width]
    result = attention_scores + bias_scores + attn_mask
    
    # Store result
    tl.store(out_ptr + linear_idx, result, mask=mask)

@triton.jit
def fused_attention_kernel(
    attention_scores_ptr,
    bias_scores_ptr,
    attn_mask_ptr,
    out_ptr,
    num_heads,
    features,
    height,
    width,
    total_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < total_elements
    
    # Each thread handles one element in the flattened attention scores [num_heads, features, height, width]
    linear_idx = offsets
    
    # Calculate coordinates within multi-dimensional tensor
    head_idx = linear_idx // (features * height * width)
    rem_idx = linear_idx % (features * height * width)
    feature_idx = rem_idx // (height * width)
    rem_idx = rem_idx % (height * width)
    h_idx = rem_idx // width
    w_idx = rem_idx % width
    
    # Load attention scores [64, 4, 144, 144]
    attention_scores = tl.load(attention_scores_ptr + linear_idx, mask=mask, other=0.0)
    
    # Load bias scores from [1, 4, 144, 144] - broadcast across heads
    # Bias is stored as [4, 144, 144], so calculate 2D coordinates
    bias_2d_offset = feature_idx * height * width + h_idx * width + w_idx
    bias_scores = tl.load(bias_scores_ptr + bias_2d_offset, mask=mask, other=0.0)
    
    # Load attention mask from [64, 144, 144] 
    mask_linear_idx = head_idx * (height * width) + h_idx * width + w_idx
    attn_mask = tl.load(attn_mask_ptr + mask_linear_idx, mask=mask, other=0.0)
    
    # Apply broadcasting: bias_scores -> [64, 4, 144, 144], attn_mask -> [64, 4, 144, 144]
    result = attention_scores + bias_scores + attn_mask
    
    # Store result
    tl.store(out_ptr + linear_idx, result, mask=mask)

@torch.fx.wrap
def fused_attention_addition(attention_scores, bias_scores, attn_mask, num_heads=64, features=4, height=144, width=144):
    total_elements = num_heads * features * height * width
    num_blocks = (total_elements + 1023) // 1024
    BLOCK_SIZE = 1024
    
    # Ensure bias_scores is in correct shape [1, features, height, width]
    if bias_scores.dim() == 3:
        # Add batch dimension
        bias_scores_2d = bias_scores.unsqueeze(0)
    elif bias_scores.dim() == 4:
        bias_scores_2d = bias_scores
    else:
        bias_scores_2d = bias_scores.view(1, features, height, width)
    
    # Create output tensor [num_heads, features, height, width]
    output = torch.empty((num_heads, features, height, width), dtype=torch.float32, device=attention_scores.device)
    
    # Launch fused kernel
    fused_attention_kernel[(num_blocks,)](
        attention_scores,
        bias_scores_2d,
        attn_mask,
        output,
        num_heads,
        features,
        height,
        width,
        total_elements,
        BLOCK_SIZE,
    )
    
    return output

def replacement_func():
    return fused_attention_addition