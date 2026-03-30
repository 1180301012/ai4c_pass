import torch
import triton
import triton.language as tl

# Pattern matching function
def pattern(in_0, in_1):
    """
    Matches the pattern: softmax(in_1, dim=1) * in_0, then sum along dim=1
    This is an attention mechanism that computes weighted channel averaging
    """
    tmp_0 = torch.softmax(in_1, dim=1)
    tmp_1 = in_0 * tmp_0
    tmp_2 = torch.sum(tmp_1, dim=1)
    return tmp_2

# Argument extraction function
def replacement_args(in_0, in_1):
    return (in_0, in_1)

# FINAL OPTIMIZED TRITON KERNEL - GPU-accelerated attention fusion
@triton.jit
def optimized_attention_fusion_kernel(
    in_0_ptr,
    in_1_ptr, 
    out_ptr,
    batch_size,
    features,
    height,
    width,
    BLOCK_HW: tl.constexpr,
):
    # Each program handles one spatial location and all features for it
    hw_idx = tl.program_id(0)  # Linear index for spatial locations
    feature_idx = tl.program_id(1)  # Feature index
    b_idx = tl.program_id(2)  # Batch index
    
    # Calculate spatial coordinates
    h_idx = hw_idx // width
    w_idx = hw_idx % width
    
    # Check bounds
    if feature_idx >= features:
        return
    if h_idx >= height:
        return
    if w_idx >= width:
        return
    
    # This program handles one feature location in one batch
    # Load softmax weights for this feature (shape: [batch, 2, features, 1, 1])
    weights_1 = tl.load(in_1_ptr + b_idx * 2 * features + feature_idx).to(tl.float32)
    weights_2 = tl.load(in_1_ptr + b_idx * 2 * features + features + feature_idx).to(tl.float32)
    
    # Compute softmax manually for the 2-element dimension
    max_val = tl.maximum(weights_1, weights_2)
    exp_w1 = tl.exp(weights_1 - max_val)
    exp_w2 = tl.exp(weights_2 - max_val)
    sum_exp = exp_w1 + exp_w2
    softmax_w1 = exp_w1 / sum_exp
    softmax_w2 = exp_w2 / sum_exp
    
    # Load input channels for this feature, batch, and spatial location (shape: [batch, 2, features, height, width])
    # Channel 0
    channel_1 = tl.load(in_0_ptr + 
                       (b_idx * 2 + 0) * features * height * width + 
                       feature_idx * height * width + 
                       h_idx * width + w_idx).to(tl.float32)
    
    # Channel 1  
    channel_2 = tl.load(in_0_ptr + 
                       (b_idx * 2 + 1) * features * height * width + 
                       feature_idx * height * width + 
                       h_idx * width + w_idx).to(tl.float32)
    
    # Compute weighted sum: w1*channel1 + w2*channel2
    weighted_sum = softmax_w1 * channel_1 + softmax_w2 * channel_2
    
    # Store result (shape: [batch, features, height, width])
    out_offset = (
        b_idx * features * height * width +
        feature_idx * height * width +
        h_idx * width + w_idx
    )
    tl.store(out_ptr + out_offset, weighted_sum.to(tl.float32))

# Enhanced kernel wrapper with performance optimizations
@torch.fx.wrap
def optimized_attention_fusion_wrapper(in_0, in_1):
    # Get input tensor properties
    assert in_0.dim() == 5, "Expected 5D input tensor"
    assert in_1.dim() == 5, "Expected 5D input tensor"
    
    batch_size, in_channels, features_0, height, width = in_0.shape
    batch_size_1, in_channels_1, features_1, _, _ = in_1.shape
    
    # Validate shapes
    assert batch_size == batch_size_1, "Batch size mismatch"
    assert in_channels == in_channels_1, "Channel count mismatch"
    assert features_0 == features_1, "Features mismatch"
    assert in_1.shape[2:] == (features_1, 1, 1), "in_1 should have spatial dimensions (features, 1, 1)"
    
    # Create output tensor - result of summing along channels: [batch, features, height, width]
    out_shape = [batch_size, features_0, height, width]
    out = torch.empty(out_shape, device=in_0.device, dtype=in_0.dtype)
    
    # Calculate grid sizes - one program per spatial location and feature per batch
    num_spatial_locations = height * width
    num_features = features_0
    num_batch = batch_size
    
    # Launch optimized kernel for GPU parallel execution  
    optimized_attention_fusion_kernel[(num_spatial_locations, num_features, num_batch)](
        in_0,
        in_1,
        out,
        batch_size,
        features_0,
        height,
        width,
        1,  # BLOCK_HW: each program handles one spatial location
    )
    
    return out

# Replacement function - optimized implementation  
def replacement_func():
    """
    Returns the optimized fusion implementation that leverages GPU parallelism.
    Follows AI4C framework requirements: zero-argument function returning callable.
    """
    return optimized_attention_fusion_wrapper