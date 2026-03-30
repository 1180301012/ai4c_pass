import torch
import triton
import triton.language as tl

@triton.jit
def multi_addition_kernel(
    activation_ptr,      # Pointer to scaled activation [1, 1, features, H, W]
    scores_ptr,         # Pointer to attention scores [batch, heads, H, W]
    mask_ptr,           # Pointer to attention mask [batch, H, W]
    output_ptr,         # Pointer to final output [batch*heads*H*W, features, H, W]
    activation_stride_0, activation_stride_1, activation_stride_2, activation_stride_3, activation_stride_4,
    scores_stride_0, scores_stride_1, scores_stride_2, scores_stride_3,
    mask_stride_0, mask_stride_1, mask_stride_2,
    output_stride_0, output_stride_1, output_stride_2, output_stride_3,
    batch_size: tl.constexpr,
    heads: tl.constexpr,
    features: tl.constexpr,
    height: tl.constexpr,
    width: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a spatial location and head
    pid = tl.program_id(0)
    total_elements = batch_size * heads * height * width
    start_idx = pid * BLOCK_SIZE
    offsets = start_idx + tl.arange(0, BLOCK_SIZE)
    mask = offsets < total_elements
    
    # Decompose index into batch, head, h, w coordinates
    flattened_idx = offsets
    w_idx = flattened_idx % width
    h_idx = (flattened_idx // width) % height
    head_idx = (flattened_idx // (width * height)) % heads
    batch_idx = flattened_idx // (width * height * heads)
    
    # Compute base indices for each tensor
    activation_idx = batch_idx * activation_stride_0  # [1, 1, features, H, W] - broadcast across batch/heads
    
    scores_idx = (batch_idx * scores_stride_0 + 
                  head_idx * scores_stride_1 + 
                  h_idx * scores_stride_2 + 
                  w_idx * scores_stride_3)
    
    mask_idx = (batch_idx * mask_stride_0 + 
                h_idx * mask_stride_1 + 
                w_idx * mask_stride_2)
    
    # Load values for this location
    activation_val = tl.load(activation_ptr + features * height * width + h_idx * width + w_idx, 
                            mask=activation_idx < batch_size * height * width, other=0.0)
    
    scores_val = tl.load(scores_ptr + scores_idx, mask=mask, other=0.0)
    mask_val = tl.load(mask_ptr + mask_idx, mask=mask, other=0.0)
    
    # Compute the optimized addition pattern
    # Original: scores + activation + mask + mask
    # Optimized: scores + activation + 2 * mask
    result = scores_val + activation_val + (2.0 * mask_val)
    
    # Store result
    tl.store(output_ptr + offsets, result, mask=mask)

@torch.fx.wrap
def optimized_multi_addition(scaled_activation, attention_scores, attention_mask, feature_dim):
    # Get dimensions based on the pattern
    batch_size, heads, height, width = attention_scores.shape[:4]
    features = feature_dim
    
    # Shape of scaled_activation: [1, 1, features, height, width]
    # We need to reshape it for broadcasting
    
    # Reshape attention scores to match final output format: [batch*heads*height*width, features, height, width]
    # This is a simplified version - actual optimization would handle the view operations more efficiently
    total_spatial = batch_size * heads * height * width
    output_shape = (total_spatial, features, height, width)
    
    output = torch.empty(output_shape, dtype=attention_scores.dtype, device=attention_scores.device)
    
    # Triton kernel launch configuration
    BLOCK_SIZE = 256
    num_programs = (total_spatial + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    multi_addition_kernel[(num_programs,)](
        scaled_activation, attention_scores, attention_mask, output,
        scaled_activation.stride(0), scaled_activation.stride(1), scaled_activation.stride(2), 
        scaled_activation.stride(3), scaled_activation.stride(4),
        attention_scores.stride(0), attention_scores.stride(1), attention_scores.stride(2), attention_scores.stride(3),
        attention_mask.stride(0), attention_mask.stride(1), attention_mask.stride(2),
        output.stride(0), output.stride(1), output.stride(2), output.stride(3),
        batch_size, heads, features, height, width,
        BLOCK_SIZE
    )
    
    return output

def pattern(scaled_activation, attention_scores, attention_mask):
    """
    Pattern: Multi-addition with scaled activation, attention scores, and attention mask
    Original pattern:
        tmp_11 = unsqueeze(0)
        tmp_12 = in_2 + tmp_11
        tmp_13 = view(1, 64, features, 64, 64)
        tmp_14 = in_3.unsqueeze(1)
        tmp_15 = tmp_14.unsqueeze(0)
        tmp_16 = tmp_13 + tmp_15
        tmp_17 = in_3.unsqueeze(1)
        tmp_18 = tmp_17.unsqueeze(0)
        tmp_19 = tmp_16 + tmp_18
    """
    # Simplified version focusing on the core mathematical operation
    # The key optimization is recognizing that mask is added twice
    tmp_11 = scaled_activation.unsqueeze(0)
    tmp_12 = attention_scores + tmp_11
    tmp_13 = tmp_12.view(1, 64, -1, 64, 64)  # -1 becomes features
    tmp_14 = attention_mask.unsqueeze(1)
    tmp_15 = tmp_14.unsqueeze(0)
    tmp_16 = tmp_13 + tmp_15
    tmp_17 = attention_mask.unsqueeze(1)
    tmp_18 = tmp_17.unsqueeze(0)
    tmp_19 = tmp_16 + tmp_18
    
    return tmp_19

def replacement_args(scaled_activation, attention_scores, attention_mask):
    # Extract feature dimension from scaled activation shape
    feature_dim = scaled_activation.shape[-1] if len(scaled_activation.shape) > 0 else 12
    return (scaled_activation, attention_scores, attention_mask)

def replacement_func():
    return optimized_multi_addition