import torch
import triton
import triton.language as tl

def pattern(input_tensor, weight, bias):
    # Layer normalization
    normalized = torch.nn.functional.layer_norm(input_tensor, (768,), weight, bias, 1e-06)
    return normalized

def replacement_args(input_tensor, weight, bias):
    return (input_tensor, weight, bias)

@triton.jit
def layernorm_kernel(
    input_ptr, weight_ptr, bias_ptr, output_ptr,
    n_rows, channels,
    BLOCK_SIZE: tl.constexpr,
):
    # Each row gets its own program
    row_idx = tl.program_id(0)
    
    # Load one row of data
    row_data = tl.load(input_ptr + row_idx * channels + tl.arange(0, BLOCK_SIZE), 
                      mask=tl.arange(0, BLOCK_SIZE) < channels, other=0.0)
    
    # Calculate mean
    mean = tl.sum(row_data) / channels
    
    # Calculate variance  
    variance = tl.sum((row_data - mean) * (row_data - mean)) / channels
    
    # Calculate inverse standard deviation
    inv_std = 1.0 / tl.sqrt(variance + 1e-06)
    
    # Load weight and bias
    weight_val = tl.load(weight_ptr + tl.arange(0, BLOCK_SIZE), 
                        mask=tl.arange(0, BLOCK_SIZE) < channels, other=0.0)
    bias_val = tl.load(bias_ptr + tl.arange(0, BLOCK_SIZE), 
                      mask=tl.arange(0, BLOCK_SIZE) < channels, other=0.0)
    
    # Apply layer normalization
    normalized = (row_data - mean) * inv_std * weight_val + bias_val
    
    # Store result
    tl.store(output_ptr + row_idx * channels + tl.arange(0, BLOCK_SIZE), 
            normalized, mask=tl.arange(0, BLOCK_SIZE) < channels)

@torch.fx.wrap
def optimized_layernorm(input_tensor, weight, bias):
    n_rows = input_tensor.shape[0]
    channels = 768  # Fixed based on the original computation
    
    output = torch.empty_like(input_tensor)
    
    BLOCK_SIZE = 512  # Power of 2
    grid = (n_rows,)
    
    layernorm_kernel[grid](
        input_ptr=input_tensor,
        weight_ptr=weight,
        bias_ptr=bias,
        output_ptr=output,
        n_rows=n_rows,
        channels=channels,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return output

def replacement_func():
    return optimized_layernorm