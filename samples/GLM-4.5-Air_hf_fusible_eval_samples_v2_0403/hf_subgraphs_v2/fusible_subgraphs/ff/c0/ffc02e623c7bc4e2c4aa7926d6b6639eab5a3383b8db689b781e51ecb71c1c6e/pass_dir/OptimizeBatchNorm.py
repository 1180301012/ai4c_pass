import torch
import triton
import triton.language as tl

def pattern(x, running_mean, running_var, weight, bias):
    """Match batch_norm with specific parameters"""
    return torch.nn.functional.batch_norm(x, running_mean, running_var, weight, bias, False, 0.1, 1e-05)

def replacement_args(x, running_mean, running_var, weight, bias):
    return (x, running_mean, running_var, weight, bias)

@triton.jit
def batch_norm_kernel(
    input_ptr,
    running_mean_ptr,
    running_var_ptr,
    weight_ptr,
    bias_ptr,
    output_ptr,
    batch_size,
    num_features,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    
    # Process a block of features
    feature_idx = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = feature_idx < num_features
    
    # Load batch norm parameters (same for all batches)
    running_mean = tl.load(running_mean_ptr + feature_idx, mask=mask, other=0.0)
    running_var = tl.load(running_var_ptr + feature_idx, mask=mask, other=0.0)
    weight = tl.load(weight_ptr + feature_idx, mask=mask, other=0.0)
    bias = tl.load(bias_ptr + feature_idx, mask=mask, other=0.0)
    
    # Pre-compute denominator for efficiency
    eps = 1e-05
    inv_std = tl.rsqrt(running_var + eps)
    
    # Process each batch
    for batch in range(batch_size):
        batch_offset = batch * num_features
        input_offsets = batch_offset + feature_idx
        
        # Load input values
        x = tl.load(input_ptr + input_offsets, mask=mask, other=0.0)
        
        # Apply batch normalization: y = (x - mean) / sqrt(var + eps) * weight + bias
        normalized = (x - running_mean) * inv_std
        result = normalized * weight + bias
        
        # Store result
        tl.store(output_ptr + input_offsets, result, mask=mask)

@torch.fx.wrap
def optimized_batch_norm(x, running_mean, running_var, weight, bias):
    batch_size = x.shape[0]
    num_features = x.shape[1] if len(x.shape) == 2 else x.shape[1]
    
    # Flatten spatial dimensions if present (for conv features)
    if len(x.shape) == 4:
        x_flat = x.reshape(batch_size * x.shape[2] * x.shape[3], num_features)
    else:
        x_flat = x.reshape(batch_size, num_features)
    
    output_flat = torch.empty_like(x_flat)
    
    BLOCK_SIZE = 256  # Number of features per program
    grid = ((num_features + BLOCK_SIZE - 1) // BLOCK_SIZE,)
    
    # Ensure contiguous memory access
    x_contiguous = x_flat.contiguous()
    running_mean_contiguous = running_mean.contiguous()
    running_var_contiguous = running_var.contiguous()
    weight_contiguous = weight.contiguous()
    bias_contiguous = bias.contiguous()
    output_contiguous = output_flat.contiguous()
    
    batch_norm_kernel[grid](
        x_contiguous,
        running_mean_contiguous,
        running_var_contiguous,
        weight_contiguous,
        bias_contiguous,
        output_contiguous,
        batch_size,
        num_features,
        BLOCK_SIZE
    )
    
    # Reshape back to original dimensions
    if len(x.shape) == 4:
        return output_flat.reshape(x.shape)
    else:
        return output_flat

def replacement_func():
    return optimized_batch_norm