import torch
import triton
import triton.language as tl

def pattern(x, normalized_shape, weight, bias, eps):
    """
    Pattern to match: just layer_norm
    """
    return torch.nn.functional.layer_norm(x, normalized_shape, weight, bias, eps)

def replacement_args(x, normalized_shape, weight, bias, eps):
    return (x, normalized_shape, weight, bias, eps)

# Use a fixed BLOCK_SIZE for simplicity - will be overridden by autotune
@triton.jit
def layer_norm_kernel(
    x_ptr, weight_ptr, bias_ptr, output_ptr,
    n_elements, hidden_size, eps,
    BLOCK_SIZE: tl.constexpr
):
    # Get row id
    row_idx = tl.program_id(0)
    
    # Compute row offset
    row_offset = row_idx * hidden_size
    
    # Create offsets for the hidden dimension
    offsets = row_offset + tl.arange(0, BLOCK_SIZE)
    mask = offsets < row_offset + hidden_size
    
    # Load the full row of data
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    
    # Compute mean using reduction - convert to float32 for precision
    x_f32 = x.to(tl.float32)
    mean_f32 = tl.sum(x_f32, axis=0) / tl.cast(hidden_size, tl.float32)
    
    # Compute variance
    diff_f32 = x_f32 - mean_f32
    var_f32 = tl.sum(diff_f32 * diff_f32, axis=0) / tl.cast(hidden_size, tl.float32)
    
    # Compute standard deviation
    rstd_f32 = 1.0 / tl.sqrt(var_f32 + eps)
    
    # Normalize in float32
    x_norm_f32 = diff_f32 * rstd_f32
    
    # Load weight and bias in float32
    w_f32 = tl.load(weight_ptr + tl.arange(0, BLOCK_SIZE), mask=mask, other=1.0).to(tl.float32)
    b_f32 = tl.load(bias_ptr + tl.arange(0, BLOCK_SIZE), mask=mask, other=0.0).to(tl.float32)
    
    # Apply affine transformation
    output_f32 = x_norm_f32 * w_f32 + b_f32
    
    # Store the result, converting back to original dtype
    output = output_f32.to(x.dtype)
    tl.store(output_ptr + offsets, output, mask=mask)

@torch.fx.wrap
def triton_layer_norm(x, normalized_shape, weight, bias, eps):
    """
    Optimized LayerNorm using Triton.
    """
    # Get hidden size from normalized_shape
    if isinstance(normalized_shape, tuple):
        hidden_size = normalized_shape[0]
    else:
        hidden_size = normalized_shape
    
    # Calculate total elements and batch size
    n_elements = x.numel()
    batch_size = n_elements // hidden_size
    
    # Create output tensor
    output = torch.empty_like(x)
    
    # Choose BLOCK_SIZE based on hidden_size
    BLOCK_SIZE = min(triton.next_power_of_2(hidden_size), 1024)
    
    # Launch kernel with batch_size blocks
    grid = (batch_size,)
    
    layer_norm_kernel[grid](
        x, weight, bias, output,
        n_elements, hidden_size, eps, BLOCK_SIZE
    )
    
    return output

def replacement_func():
    return triton_layer_norm