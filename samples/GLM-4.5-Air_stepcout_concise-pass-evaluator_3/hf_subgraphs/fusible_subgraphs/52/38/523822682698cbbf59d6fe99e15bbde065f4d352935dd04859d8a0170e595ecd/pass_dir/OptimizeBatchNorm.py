import torch
import triton
import triton.language as tl

# Pattern matching function
def pattern(in_7, tmp_0, tmp_1, tmp_3, tmp_2):
    """ Match the batch normalization operation """
    tmp_7 = torch.nn.functional.batch_norm(in_7, tmp_0, tmp_1, tmp_3, tmp_2, False, 0.1, 1e-05)
    return tmp_7

# Argument extraction function
def replacement_args(in_7, tmp_0, tmp_1, tmp_3, tmp_2):
    return (in_7, tmp_0, tmp_1, tmp_3, tmp_2)

# Optimized kernel for batch normalization
@triton.jit
def batch_norm_kernel(
    x_ptr,
    running_mean_ptr,
    running_var_ptr,
    weight_ptr,
    bias_ptr,
    out_ptr,
    batch_size,
    num_features,
    BLOCK_SIZE: tl.constexpr,
):
    """ Triton kernel optimized for batch normalization (inference mode) """
    # Program ID
    pid = tl.program_id(0)
    
    # Range for this program
    x_start = pid * BLOCK_SIZE
    x_offsets = x_start + tl.arange(0, BLOCK_SIZE)
    
    # Create mask for input elements
    mask = x_offsets < (batch_size * num_features)
    
    # Load input data efficiently
    in_flat = tl.load(x_ptr + x_offsets, mask=mask, other=0.0)
    
    # Load batch normalization parameters
    running_mean = tl.load(running_mean_ptr, mask=tl.arange(0, num_features) < num_features, other=0.0)
    running_var = tl.load(running_var_ptr, mask=tl.arange(0, num_features) < num_features, other=0.0)
    weight = tl.load(weight_ptr, mask=tl.arange(0, num_features) < num_features, other=1.0)
    bias = tl.load(bias_ptr, mask=tl.arange(0, num_features) < num_features, other=0.0)
    
    # Reshape for broadcasting
    in_reshaped = in_flat.reshape((batch_size, num_features))
    
    # Batch normalization formula: (in - running_mean) / sqrt(running_var + eps) * weight + bias
    epsilon = 1e-05
    
    # Vectorized computation using Triton operations
    sub = in_reshaped - running_mean
    sqrt_var = tl.sqrt(running_var + epsilon)
    normalized = sub / sqrt_var
    scaled = normalized * weight
    result = scaled + bias
    
    # Flatten back to 1D and store
    out_flat = result.flatten()
    tl.store(out_ptr + x_offsets, out_flat, mask=mask)

# Simplified optimized kernel for batch normalization
@triton.jit
def batch_norm_kernel_simple(
    x_ptr,
    running_mean_ptr,
    running_var_ptr,
    weight_ptr,
    bias_ptr,
    out_ptr,
    batch_size: tl.constexpr,
    num_features: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """ Triton kernel optimized for batch normalization with proper constexpr parameters """
    # Use 2D program IDs for batch and feature dimensions
    pid_batch = tl.program_id(0)
    pid_feature = tl.program_id(1)
    
    # Check if this batch element is valid
    if pid_batch >= batch_size:
        return
    
    # Calculate grid for feature dimension
    grid_f = (num_features + BLOCK_SIZE - 1) // BLOCK_SIZE
    if pid_feature >= grid_f:
        return
    
    # Range for current feature block (BLOCK_SIZE is now constexpr)
    feat_start = pid_feature * BLOCK_SIZE
    feat_offsets = feat_start + tl.arange(0, BLOCK_SIZE)
    
    # Masks for bounds checking
    feat_mask = feat_offsets < num_features
    
    # Calculate total offset in flattened array
    batch_offset = pid_batch * num_features
    total_offsets = batch_offset + feat_offsets
    total_mask = total_offsets < (batch_size * num_features)
    
    # Load input data
    x = tl.load(x_ptr + total_offsets, mask=total_mask, other=0.0)
    
    # Load batch norm parameters for current feature block
    running_mean = tl.load(running_mean_ptr + feat_offsets,
                         mask=feat_mask,
                         other=0.0)
    running_var = tl.load(running_var_ptr + feat_offsets,
                        mask=feat_mask,
                        other=1.0)
    weight = tl.load(weight_ptr + feat_offsets,
                   mask=feat_mask,
                   other=1.0)
    bias_val = tl.load(bias_ptr + feat_offsets,
                     mask=feat_mask,
                     other=0.0)
    
    # Batch normalization computation: (x - running_mean) / sqrt(running_var + eps) * weight + bias
    epsilon = 1e-05
    
    # Vectorized computation
    sub = x - running_mean
    sqrt_var = tl.sqrt(running_var + epsilon)
    normalized = sub / sqrt_var
    scaled = normalized * weight
    result = scaled + bias_val
    
    # Store result
    tl.store(out_ptr + total_offsets, result, mask=total_mask)

# Kernel wrapper
@torch.fx.wrap
def optimized_batch_norm(in_7, tmp_0, tmp_1, tmp_3, tmp_2):
    """ Optimized batch normalization using Triton """
    batch_size, num_features = in_7.shape
    
    # Use power-of-2 block size (128 is a good default)
    BLOCK_SIZE = 128
    
    # Calculate 2D grid size: one dimension for batch, another for features
    grid_m = batch_size
    grid_n = (num_features + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Allocate output tensor
    out = torch.empty_like(in_7)
    
    # Launch kernel with 2D grid and all constexpr parameters
    batch_norm_kernel_simple[(grid_m, grid_n)](
        x_ptr=in_7,
        running_mean_ptr=tmp_0,
        running_var_ptr=tmp_1,
        weight_ptr=tmp_3,
        bias_ptr=tmp_2,
        out_ptr=out,
        batch_size=batch_size,
        num_features=num_features,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out

# Replacement function
def replacement_func():
    return optimized_batch_norm