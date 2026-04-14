import torch
import triton
import triton.language as tl

def pattern(x, scaling_param):
    """Pattern matching the normalize+scale computation sequence"""
    # ReLU activation
    relu_out = torch.nn.functional.relu(x, inplace=True)
    
    # Flatten starting from dimension 2
    flattened = torch.flatten(relu_out, 2)
    
    # L2 norm along last dimension, keeping dim for broadcasting
    norm_vals = torch.functional.norm(flattened, dim=-1, keepdim=True)
    
    # Scale the norm by a constant
    scaled_norm = norm_vals * scaling_param
    
    # Clamp to avoid division by zero
    clamped = scaled_norm.clamp(min=1e-05)
    
    # Normalize the flattened tensor
    normalized = flattened / clamped
    
    # Scale by the parameter
    result = normalized * scaling_param
    
    return result

def replacement_args(x, scaling_param):
    """Extract arguments needed for the fused kernel"""
    return (x, scaling_param)

@triton.jit
def fused_normalize_scale_kernel(
    x_ptr,
    scaling_param_ptr,
    out_ptr,
    input_shape_1,  # size after dimension 1
    input_shape_2,  # size after dimension 2
    input_shape_3,  # size after dimension 3
    scaling_param_f32,
    BLOCK_SIZE: tl.constexpr,
):
    """Fused kernel for normalization and scaling operations"""
    # Calculate total elements per feature (after flattening)
    elements_per_feature = input_shape_2 * input_shape_3
    
    # Each program handles one feature vector from the flattened tensor
    batch_idx = tl.program_id(0)
    channel_idx = tl.program_id(1)
    feature_idx = tl.program_id(2) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    
    # Output indices in the flattened tensor
    flattened_idx = batch_idx * input_shape_1 * elements_per_feature + \
                   channel_idx * elements_per_feature + feature_idx
    
    # Check bounds to avoid memory access violations
    mask = feature_idx < elements_per_feature
    
    # Load input data
    input_data = tl.load(x_ptr + flattened_idx, mask=mask, other=0.0)
    
    # Apply ReLU activation
    relu_out = tl.maximum(input_data, 0.0)
    
    # Compute L2 norm for this feature vector
    # Sum of squares across all elements in this feature
    sum_of_squares = tl.sum(relu_out * relu_out)
    norm_val = tl.sqrt(sum_of_squares + 1e-20)  # Add small epsilon for numerical stability
    
    # Scale the norm
    scaled_norm = norm_val * scaling_param_f32
    
    # Clamp the norm
    clamped_norm = tl.maximum(scaled_norm, 1e-05)
    
    # Normalize the feature vector and scale by parameter
    scale_factor = scaling_param_f32 / clamped_norm
    result = relu_out * scale_factor
    
    # Store the result
    tl.store(out_ptr + flattened_idx, result, mask=mask)

@torch.fx.wrap
def fused_normalize_scale(x, scaling_param):
    """Wrapper for the fused normalization and scaling kernel"""
    input_shape = x.shape
    
    # Determine parameters for flattening (starting from dimension 2)
    batch_size = input_shape[0]
    channels = input_shape[1]
    spatial_size = input_shape[2] * input_shape[3] if len(input_shape) > 3 else 1
    
    # Total elements in the flattened tensor
    total_elements = batch_size * channels * spatial_size
    elements_per_feature = spatial_size
    
    # Set up grid dimensions
    # We organize the computation by: batch -> channel -> feature positions
    num_batches = batch_size
    num_channels = channels
    num_features_per_channel = (elements_per_feature + 1023) // 1024  # Block size of 1024
    total_features = num_channels * num_features_per_channel
    
    # Create output tensor
    out = torch.empty_like(x)
    
    # Calculate grid for launching kernel
    grid = (num_batches, num_channels, num_features_per_channel)
    
    # Launch the fused kernel
    fused_normalize_scale_kernel[grid](
        x_ptr=x,
        scaling_param_ptr=None,  # We'll pass the scalar value directly
        out_ptr=out,
        input_shape_1=channels,
        input_shape_2=input_shape[2],
        input_shape_3=input_shape[3],
        scaling_param_f32=float(scaling_param.item()),
        BLOCK_SIZE=1024,
    )
    
    return out

def replacement_func():
    """Returns the fused implementation"""
    return fused_normalize_scale