import torch
import triton
import triton.language as tl

def pattern(in_2, normalized_shape, weight, bias, eps):
    """
    Pattern: LayerNorm
    """
    result = torch.nn.functional.layer_norm(in_2, normalized_shape, weight, bias, eps)
    return result

def replacement_args(in_2, normalized_shape, weight, bias, eps):
    return (in_2, normalized_shape, weight, bias, eps)

@triton.jit
def layer_norm_kernel(
    input_ptr,
    weight_ptr,
    bias_ptr,
    output_ptr,
    N,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Optimized LayerNorm kernel
    Each program handles one row of the input
    """
    row_idx = tl.program_id(0)
    
    # Calculate row offset
    row_start = row_idx * N
    offsets = row_start + tl.arange(0, BLOCK_SIZE)
    mask = tl.arange(0, BLOCK_SIZE) < N
    
    # Load input data
    x = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    
    # Calculate mean
    mean = tl.sum(x, axis=0) / N
    
    # Calculate variance
    x_centered = tl.where(mask, x - mean, 0.0)
    variance = tl.sum(x_centered * x_centered, axis=0) / N
    
    # Normalize
    rstd = 1.0 / tl.sqrt(variance + eps)
    x_normalized = x_centered * rstd
    
    # Load weight and bias
    weight = tl.load(weight_ptr + tl.arange(0, BLOCK_SIZE), mask=mask, other=0.0)
    bias = tl.load(bias_ptr + tl.arange(0, BLOCK_SIZE), mask=mask, other=0.0)
    
    # Apply affine transformation
    output = tl.where(mask, x_normalized * weight + bias, 0.0)
    
    # Store output
    tl.store(output_ptr + offsets, output, mask=mask)

@torch.fx.wrap
def optimized_layer_norm(input_tensor, normalized_shape, weight, bias, eps):
    """
    Optimized LayerNorm implementation using Triton
    """
    # Get dimensions
    batch_size = input_tensor.shape[0]
    seq_len = input_tensor.shape[1]
    hidden_dim = normalized_shape[0]
    
    # Flatten batch and sequence dimensions
    input_2d = input_tensor.view(-1, hidden_dim)
    num_rows = input_2d.shape[0]
    
    # Allocate output
    output = torch.empty_like(input_2d)
    
    # Find next power of 2 for BLOCK_SIZE
    BLOCK_SIZE = triton.next_power_of_2(hidden_dim)
    
    # Launch kernel
    grid = (num_rows,)
    layer_norm_kernel[grid](
        input_2d,
        weight,
        bias,
        output,
        hidden_dim,
        eps,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    # Reshape back to original shape
    return output.view(batch_size, seq_len, hidden_dim)

def replacement_func():
    return optimized_layer_norm