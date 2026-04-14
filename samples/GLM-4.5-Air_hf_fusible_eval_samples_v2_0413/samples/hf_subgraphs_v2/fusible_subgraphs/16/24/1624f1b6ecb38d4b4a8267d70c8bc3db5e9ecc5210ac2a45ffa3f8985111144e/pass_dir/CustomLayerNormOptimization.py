import torch
import triton
import triton.language as tl

# Pattern matching function - matches layer normalization
def pattern(input_tensor, weight, bias):
    # Dropout with 0.0 rate is a no-op, so we directly apply layer norm
    normalized = torch.nn.functional.layer_norm(input_tensor, (768,), bias, weight, 1e-06)
    # Return both input (for dropout bypass) and normalized result
    return input_tensor, normalized

# Argument extraction function
def replacement_args(input_tensor, weight, bias):
    return (input_tensor, weight, bias)

# Optimized Triton kernel for layer normalization
@triton.jit
def layernorm_kernel(
    input_ptr, weight_ptr, bias_ptr, output_ptr,
    n_elements, channels,
    BLOCK_SIZE: tl.constexpr, EPM: tl.constexpr,
):
    # Each row gets its own program
    row_idx = tl.program_id(0)
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < channels
    
    # Load data for this row
    row_data = tl.load(input_ptr + row_idx * n_elements + col_offsets, mask=mask, other=0.0)
    
    # Calculate mean
    mean = tl.sum(row_data, axis=0) / channels
    
    # Calculate variance
    centered = row_data - mean
    variance = tl.sum(centered * centered, axis=0) / channels
    
    # Layer normalization formula
    inv_std = 1.0 / tl.sqrt(variance + 1e-06)
    
    # Load weight and bias
    weight_val = tl.load(weight_ptr + col_offsets, mask=mask, other=0.0)
    bias_val = tl.load(bias_ptr + col_offsets, mask=mask, other=0.0)
    
    # Apply normalization
    normalized = (row_data - mean) * inv_std * weight_val + bias_val
    
    # Store result
    tl.store(output_ptr + row_idx * channels + col_offsets, normalized, mask=mask)

@torch.fx.wrap
def optimized_layernorm(input_tensor, weight, bias):
    # Get tensor shapes
    n_rows = input_tensor.shape[0]
    channels = 768  # Hardcoded based on the original computation
    
    # Output tensor
    output = torch.empty_like(input_tensor)
    
    # Triton kernel launch configuration
    BLOCK_SIZE = 256
    EPM = 8  # Elements per memory operation
    num_programs = n_rows
    
    # Launch kernel
    layernorm_kernel[(num_programs,)](
        input_ptr=input_tensor,
        weight_ptr=weight,
        bias_ptr=bias,
        output_ptr=output,
        n_elements=input_tensor.shape[1],
        channels=channels,
        BLOCK_SIZE=BLOCK_SIZE,
        EPM=EPM,
    )
    
    return input_tensor, output  # Return original (for dropout) + normalized

# Replacement function
def replacement_func():
    return optimized_layernorm