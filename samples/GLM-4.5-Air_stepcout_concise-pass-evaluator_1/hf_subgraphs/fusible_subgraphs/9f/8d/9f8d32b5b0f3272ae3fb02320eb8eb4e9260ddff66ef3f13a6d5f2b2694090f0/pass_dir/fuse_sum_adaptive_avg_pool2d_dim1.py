import torch
import triton
import triton.language as tl

def pattern(in_0):
    """
    Pattern matching: sum along dim=1 followed by adaptive_avg_pool2d with output size 1
    """
    tmp_0 = in_0.sum(dim=1)
    tmp_1 = torch.nn.functional.adaptive_avg_pool2d(tmp_0, 1)
    return tmp_1

def replacement_args(in_0):
    """
    Extract arguments needed for the replacement kernel
    """
    return (in_0,)

@triton.jit
def fused_sum_avg_pool_kernel(
    input_ptr,
    output_ptr,
    n_batch,
    n_prev_features,
    n_spatial_features,
    height,
    width,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fused kernel that combines sum along dim=1 and adaptive average pooling (output size 1x1)
    """
    # Each program handles one output feature for a batch
    pid = tl.program_id(0)
    
    # Calculate batch and feature indices
    batch_idx = pid // n_spatial_features
    spatial_feature_idx = pid % n_spatial_features
    
    # Check if we're within valid range
    if batch_idx >= n_batch:
        return
    
    # Initialize accumulator for this (batch, spatial_feature) combination
    spatial_elements = height * width
    accumulator = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    
    # Process spatial dimensions in blocks
    for i in range(0, spatial_elements, BLOCK_SIZE):
        offset = i + tl.arange(0, BLOCK_SIZE)
        mask = offset < spatial_elements
        
        # Calculate spatial coordinates
        spatial_h = offset // width
        spatial_w = offset % width
        
        # 5D index calculation: [batch, prev_feat_idx, spatial_feat_idx, h, w]
        # We sum along prev_feat_idx (dim=1), so we need to accumulate over prev_feat_idx
        for prev_feat_idx in range(n_prev_features):
            flat_input_idx = (
                batch_idx * n_prev_features * n_spatial_features * spatial_elements +
                prev_feat_idx * n_spatial_features * spatial_elements +
                spatial_feature_idx * spatial_elements +
                spatial_h * width + spatial_w
            )
            
            # Load and accumulate with bounds checking
            input_vals = tl.load(input_ptr + flat_input_idx, mask=mask, other=0.0)
            accumulator += input_vals
    
    # Sum the block and compute average
    block_sum = tl.sum(accumulator, axis=0)
    result = block_sum / (n_prev_features * spatial_elements)
    
    # Store result in output [batch_size, n_spatial_features, 1, 1]
    output_idx = batch_idx * n_spatial_features + spatial_feature_idx
    tl.store(output_ptr + output_idx, result)

@torch.fx.wrap
def fused_sum_adaptive_avg_pool2d(input_tensor):
    """
    Wrapper function that launches the fused kernel
    """
    # Input has shape [batch_size, n_prev_features, n_spatial_features, height, width]
    if len(input_tensor.shape) != 5:
        raise ValueError(f"Expected 5D input, got {len(input_tensor.shape)}D")
    
    batch_size, n_prev_features, n_spatial_features, height, width = input_tensor.shape
    
    # Output shape should be [batch_size, n_spatial_features, 1, 1]
    output_shape = (batch_size, n_spatial_features, 1, 1)
    output = torch.empty(output_shape, dtype=input_tensor.dtype, device=input_tensor.device)
    
    # Grid configuration: one program per (batch, spatial_feature) combination
    grid_size = batch_size * n_spatial_features
    
    # Launch the kernel
    fused_sum_avg_pool_kernel[(grid_size,)](
        input_ptr=input_tensor,
        output_ptr=output,
        n_batch=batch_size,
        n_prev_features=n_prev_features,
        n_spatial_features=n_spatial_features,
        height=height,
        width=width,
        BLOCK_SIZE=1024,
    )
    
    return output

def replacement_func():
    """
    Returns the replacement function (must be zero-argument)
    """
    return fused_sum_adaptive_avg_pool2d