import torch

# Simple pattern to start with - just match multiplication and sum
def pattern(tensor_group, input_features):
    """Simple pattern for multiplication and sum"""
    weighted = tensor_group * input_features
    result = torch.sum(weighted, dim=1)
    return (result,)

# Argument extraction function
def replacement_args(tensor_group, input_features):
    return (tensor_group, input_features)

# Optimized Triton kernel for the entire fusion
@triton.jit
def fused_softmax_multiply_sum_kernel(
    attention_scores_ptr,
    input_features_ptr,
    output_ptr,
    batch_size,
    num_groups,
    channels,
    spatial_height,
    spatial_width,
    BLOCK_M: tl.constexpr,  # Block size for matrix operations
    BLOCK_N: tl.constexpr,  # Block size for channels
):
    """Fused kernel that performs softmax + reshape + multiply + sum"""
    
    pid = tl.program_id(0)
    
    # Only process valid batch elements
    if pid >= batch_size:
        return
    
    # Process one batch at a time
    batch_offset = pid * num_groups * channels
    spatial_size = spatial_height * spatial_width
    
    # Load attention scores for this batch
    attention_scores = tl.load(
        attention_scores_ptr + batch_offset,
        mask=True
    )
    
    # Apply softmax using Triton operations
    max_score = tl.max(attention_scores, axis=0)
    exp_scores = tl.exp(attention_scores - max_score)
    sum_exp = tl.sum(exp_scores, axis=0)
    softmax_weights = exp_scores / sum_exp
    
    # Reshape weights to [num_groups, channels] for broadcasting
    weights_reshaped = softmax_weights
    
    # Process each group and channel combination
    for gid in range(num_groups):
        for cid in range(channels):
            weight_val = weights_reshaped[gid * channels + cid]
            
            # Load input features for this batch, group, channel
            feat_offset = (pid * num_groups * channels * spatial_size + 
                          gid * channels * spatial_size + 
                          cid * spatial_size)
            
            # Load and multiply spatial features efficiently
            for hid in range(0, spatial_height, BLOCK_M):
                for wid in range(0, spatial_width, BLOCK_N):
                    # Calculate memory offsets for spatial block
                    sp_offset = feat_offset + hid * spatial_width + wid
                    in_block = tl.load(
                        input_features_ptr + sp_offset,
                        mask=(hid < spatial_height) & (wid < spatial_width),
                        other=0.0
                    )
                    
                    # Multiply and accumulate spatial features
                    weighted_block = in_block * weight_val
                    
                    # Store weighted features
                    tl.store(
                        output_ptr + sp_offset,
                        weighted_block,
                        mask=(hid < spatial_height) & (wid < spatial_width)
                    )

@torch.fx.wrap
def fused_softmax_multiply_sum(attention_scores, input_features):
    """Wrapper function for the fused kernel"""
    
    # Get input shapes
    batch_size = attention_scores.shape[0]
    num_groups = attention_scores.shape[1]
    channels = attention_scores.shape[3]  # Last dimension before reshape
    spatial_height = input_features.shape[3]
    spatial_width = input_features.shape[4]
    
    # Create output tensor
    output_shape = [batch_size, channels, spatial_height, spatial_width]
    output = torch.empty(output_shape, dtype=input_features.dtype, device=input_features.device)
    
    # Set up Triton launch configuration
    BLOCK_M = 16
    BLOCK_N = 16
    grid_size = batch_size
    
    # Launch the kernel
    fused_softmax_multiply_sum_kernel[grid_size](
        attention_scores,
        input_features, 
        output,
        batch_size,
        num_groups,
        channels,
        spatial_height,
        spatial_width,
        BLOCK_M,
        BLOCK_N
    )
    
    return output

# Replacement function (returns the fused kernel function)
def replacement_func():
    return fused_softmax_multiply_sum