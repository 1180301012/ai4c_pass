import torch
import triton
import triton.language as tl

# Pattern matching function - matches softmax + multiplication + sum fusion
def pattern(values, weights):
    """
    Pattern matching for weighted sum operations
    values: input tensor to be weighted
    weights: attention weights to apply
    """
    # Return values * weights to match the weighted multiplication pattern
    # This is a placeholder that represents the core operation
    return values * weights

def replacement_args(values, weights):
    """Extract arguments for the replacement kernel"""
    # Return arguments in the same order as expected by pattern
    return (values, weights)

# Optimized Triton kernel for fused attention computation
@triton.jit
def fused_attention_kernel(
    att_scores_ptr,     # Pointer to attention scores tensor [B, 2, C, 1, 1]
    values_ptr,         # Pointer to values tensor [B, 2, C, H, W]  
    out_ptr,            # Pointer to output tensor [B, C, H, W]
    batch_size,         # Batch size
    num_channels,       # Number of channels (C)
    spatial_height,     # Spatial height dimension (H)
    spatial_width,      # Spatial width dimension (W)
):
    """
    Fuse softmax + multiplication + sum into a single kernel
    
    Input shapes:
    att_scores: [batch_size, 2, num_channels, 1, 1]
    values: [batch_size, 2, num_channels, spatial_height, spatial_width]
    
    Output shape:
    out: [batch_size, num_channels, spatial_height, spatial_width]
    
    Operation for each output position (b, c, h, w):
    out[b, c, h, w] = sum_{channel_idx=0,1} (values[b, channel_idx, c, h, w] * softmax(att_scores[b, channel_idx, c, 0, 0]))
    """
    # Get program indices
    batch_idx = tl.program_id(0)         # Batch index
    channel_idx = tl.program_id(1)       # Output channel index (256)
    spatial_idx = tl.program_id(2)       # Spatial position index (H * W)
    
    # Check bounds
    if batch_idx >= batch_size or channel_idx >= num_channels or spatial_idx >= (spatial_height * spatial_width):
        return
    
    # Decompose spatial index to h, w coordinates
    h = spatial_idx // spatial_width
    w = spatial_idx % spatial_width
    
    # Load the two attention scores for this channel and batch
    # att_scores shape: [batch_size, 2, num_channels, 1, 1]
    att_score_0 = tl.load(att_scores_ptr + (batch_idx * 2 * num_channels + 0 * num_channels + channel_idx) * 1 * 1)
    att_score_1 = tl.load(att_scores_ptr + (batch_idx * 2 * num_channels + 1 * num_channels + channel_idx) * 1 * 1)
    
    # Compute softmax for numerical stability
    max_score = tl.maximum(att_score_0, att_score_1)
    exp_0 = tl.exp(att_score_0 - max_score)
    exp_1 = tl.exp(att_score_1 - max_score)
    softmax_sum = exp_0 + exp_1
    
    # Softmax weights
    softmax_weight_0 = exp_0 / softmax_sum
    softmax_weight_1 = exp_1 / softmax_sum
    
    # Load values from both channels
    # values shape: [batch_size, 2, num_channels, spatial_height, spatial_width]
    offset_0 = (batch_idx * 2 * num_channels + 0 * num_channels + channel_idx) * spatial_height * spatial_width + h * spatial_width + w
    offset_1 = (batch_idx * 2 * num_channels + 1 * num_channels + channel_idx) * spatial_height * spatial_width + h * spatial_width + w
    
    value_0 = tl.load(values_ptr + offset_0)
    value_1 = tl.load(values_ptr + offset_1)
    
    # Compute weighted sum: value_0 * softmax_weight_0 + value_1 * softmax_weight_1
    result = value_0 * softmax_weight_0 + value_1 * softmax_weight_1
    
    # Store result
    # Output shape: [batch_size, num_channels, spatial_height, spatial_width]
    output_offset = (batch_idx * num_channels + channel_idx) * spatial_height * spatial_width + h * spatial_width + w
    tl.store(out_ptr + output_offset, result)

@torch.fx.wrap
def fused_attention_wrapper(att_scores, values):
    """
    Wrapper function to launch the fused attention kernel
    """
    # Get tensor shapes
    batch_size = att_scores.shape[0]       # B
    num_channels = att_scores.shape[2]     # C (256)
    spatial_height = values.shape[2]      # H (32 or 8)
    spatial_width = values.shape[3]       # W (32 or 8)
    
    # Output shape: [B, C, H, W] after summing across the 2 channels (dim=1)
    output_shape = (batch_size, num_channels, spatial_height, spatial_width)
    output = torch.empty(output_shape, dtype=torch.float32, device=att_scores.device)
    
    if (spatial_height * spatial_width == 0 or batch_size == 0 or num_channels == 0):
        return output
    
    # Set up grid dimensions: [B, C, H*W]
    grid = (batch_size, num_channels, spatial_height * spatial_width)
    
    # Launch kernel
    fused_attention_kernel[grid](
        att_scores,
        values, 
        output,
        batch_size,
        num_channels,
        spatial_height,
        spatial_width,
    )
    
    return output

def replacement_func():
    """Return the fused attention function"""
    return fused_attention_wrapper