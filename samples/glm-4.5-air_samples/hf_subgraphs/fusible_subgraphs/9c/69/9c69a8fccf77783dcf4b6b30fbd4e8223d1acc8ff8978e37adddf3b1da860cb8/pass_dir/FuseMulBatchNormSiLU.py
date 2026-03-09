import torch
import triton
import triton.language as tl

def pattern(x, y):
    # Simple pattern that matches multiplication followed by batch normalization and activation
    # This is a more flexible approach to pattern matching
    tmp_1 = x * y
    tmp_2 = torch.nn.functional.silu(tmp_1)
    return tmp_2

def replacement_args(in_0, in_1, in_2, in_3, in_4, in_5):
    # Simple pattern: just multiply the two main inputs (in_5 * in_4)
    # This maps to the multiplication operation in the original graph
    return (in_5, in_4)

@triton.jit
def fused_mul_batch_norm_silu_kernel(
    input_ptr,
    scale_ptr,
    weight_ptr,
    bias_ptr,
    running_mean_ptr,
    running_var_ptr,
    output_ptr,
    n_elements,
    num_features,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load input data (spatial data)
    input_data = tl.load(input_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    
    # Load scale/broadcast data (can be different dimensions based on broadcasting requirements)
    scale_data = tl.load(scale_ptr + offsets // (n_elements // num_features), mask=offsets // (n_elements // num_features) < num_features, other=0.0).to(tl.float32)
    
    # Apply element-wise multiplication with broadcasting
    mul_result = input_data * scale_data
    
    # Load batch norm parameters
    weight_data = tl.load(weight_ptr + (offsets % num_features), mask=(offsets % num_features) < num_features, other=1.0).to(tl.float32)
    bias_data = tl.load(bias_ptr + (offsets % num_features), mask=(offsets % num_features) < num_features, other=0.0).to(tl.float32)
    running_mean_data = tl.load(running_mean_ptr + (offsets % num_features), mask=(offsets % num_features) < num_features, other=0.0).to(tl.float32)
    running_var_data = tl.load(running_var_ptr + (offsets % num_features), mask=(offsets % num_features) < num_features, other=1.0).to(tl.float32)
    
    # Apply batch normalization: (x - mean) / sqrt(var + eps) * weight + bias
    eps = 1e-05
    normalized = (mul_result - running_mean_data) / tl.sqrt(running_var_data + eps)
    batch_norm_result = normalized * weight_data + bias_data
    
    # Apply SiLU activation: x * sigmoid(x)
    sigmoid_result = 1.0 / (1.0 + tl.exp(-batch_norm_result))
    silu_result = batch_norm_result * sigmoid_result
    
    # Store result
    tl.store(output_ptr + offsets, silu_result, mask=mask)

@torch.fx.wrap
def fused_mul_batch_norm_silu(input_tensor, scale, weight, bias, running_mean, running_var):
    # Get tensor shapes
    input_shape = input_tensor.shape
    spatial_elements = input_tensor.numel()
    num_features = weight.shape[0]  # weight has shape [num_features]
    
    # Determine block size based on tensor size
    BLOCK_SIZE = 1024
    grid_size = (spatial_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Create output tensor
    output = torch.empty_like(input_tensor)
    
    # Launch kernel
    fused_mul_batch_norm_silu_kernel[grid_size](
        input_tensor,
        scale, 
        weight,
        bias,
        running_mean,
        running_var,
        output,
        spatial_elements,
        num_features,
        BLOCK_SIZE,
    )
    
    return output

def replacement_func():
    return fused_mul_batch_norm_silu