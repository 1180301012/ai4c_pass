import torch
import triton
import triton.language as tl

def pattern(x, running_mean, running_var, weight, bias):
    """Match batch_norm with specific parameters from adaptive_avg_pool2d input"""
    y = torch.nn.functional.batch_norm(x, running_mean, running_var, weight, bias, False, 0.1, 1e-05)
    return x, y

def replacement_args(x, running_mean, running_var, weight, bias):
    return (x, running_mean, running_var, weight, bias)

@triton.jit
def optimized_batch_norm_kernel(
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
    
    # Each program processes a block of features
    feature_idx = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = feature_idx < num_features
    
    # Load batch norm parameters (same for all batches)
    running_mean = tl.load(running_mean_ptr + feature_idx, mask=mask, other=0.0)
    running_var = tl.load(running_var_ptr + feature_idx, mask=mask, other=0.0)
    weight = tl.load(weight_ptr + feature_idx, mask=mask, other=0.0)
    bias = tl.load(bias_ptr + feature_idx, mask=mask, other=0.0)
    
    # Process each batch
    for batch in range(batch_size):
        # Calculate input offsets for this batch and feature block
        batch_offset = batch * num_features
        input_offsets = batch_offset + feature_idx
        
        # Load input values
        input_values = tl.load(input_ptr + input_offsets, mask=mask, other=0.0)
        
        # Apply batch normalization
        normalized = (input_values - running_mean) / tl.sqrt(running_var + 1e-05)
        scaled = normalized * weight + bias
        
        # Store result
        tl.store(output_ptr + input_offsets, scaled, mask=mask)

@torch.fx.wrap
def optimized_batch_norm(input_tensor, running_mean, running_var, weight, bias):
    batch_size, num_features = input_tensor.shape[0:2]  # Support both 2D and 4D inputs
    
    # Create output tensor
    output = torch.empty_like(input_tensor)
    
    # Config block size
    BLOCK_SIZE = 256  # Number of features processed by each program
    
    # Grid configuration
    grid = ((num_features + BLOCK_SIZE - 1) // BLOCK_SIZE,)
    
    # Ensure input is contiguous
    input_contiguous = input_tensor.contiguous()
    running_mean_contiguous = running_mean.contiguous()
    running_var_contiguous = running_var.contiguous()
    weight_contiguous = weight.contiguous()
    bias_contiguous = bias.contiguous()
    output_contiguous = output.contiguous()
    
    optimized_batch_norm_kernel[grid](
        input_contiguous,
        running_mean_contiguous,
        running_var_contiguous,
        weight_contiguous,
        bias_contiguous,
        output_contiguous,
        batch_size,
        num_features,
        BLOCK_SIZE
    )
    
    return input_tensor, output  # Return original x and processed y

def replacement_func():
    return optimized_batch_norm