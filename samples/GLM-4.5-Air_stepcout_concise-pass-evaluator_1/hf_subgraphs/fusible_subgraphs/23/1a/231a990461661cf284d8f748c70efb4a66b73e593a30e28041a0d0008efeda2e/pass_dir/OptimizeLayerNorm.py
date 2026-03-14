import torch
import triton
import triton.language as tl

def pattern(x, weight, bias, normalized_shape):
    return torch.nn.functional.layer_norm(x, normalized_shape, weight, bias, 1e-05)

def replacement_args(x, weight, bias, normalized_shape):
    return (x, weight, bias, normalized_shape)

@triton.jit
def layernorm_kernel(
    x_ptr,
    weight_ptr,
    bias_ptr,
    y_ptr,
    n_elements,
    hidden_size,
    eps: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles one row of the input tensor
    row_idx = tl.program_id(0)
    row_offset = row_idx * n_elements
    
    # Create offset masks
    offsets = row_offset + tl.arange(0, BLOCK_SIZE)
    mask = offsets < (row_idx + 1) * n_elements
    
    # Load the row
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    
    # Compute mean
    mean = tl.sum(x, axis=0) / n_elements
    
    # Compute variance
    x_centered = x - mean
    x_mean_sq = x_centered * x_centered
    var = tl.sum(x_mean_sq, axis=0) / n_elements
    
    # Compute std
    std = tl.sqrt(var + eps)
    
    # Normalize
    y_norm = x_centered / std
    
    # Load scale and bias
    weight = tl.load(weight_ptr + tl.arange(0, hidden_size), mask=(tl.arange(0, hidden_size) < hidden_size))
    bias = tl.load(bias_ptr + tl.arange(0, hidden_size), mask=(tl.arange(0, hidden_size) < hidden_size))
    
    # Apply affine transformation
    y = y_norm * weight + bias
    
    # Store result
    tl.store(y_ptr + offsets, y, mask=mask)

@torch.fx.wrap
def optimized_layernorm(x, weight, bias, normalized_shape):
    """
    Optimized LayerNorm implementation using Triton.
    """
    # Ensure normalized_shape is a tuple
    if isinstance(normalized_shape, int):
        normalized_shape = (normalized_shape,)
    
    # Get the last dimension from normalized_shape
    hidden_size = normalized_shape[-1] if len(normalized_shape) > 0 else x.shape[-1]
    
    # Handle 2D input (most common case): [batch_size, seq_len, hidden_size]
    if x.dim() == 3:
        batch_size, seq_len, _ = x.shape
        n_elements_per_row = seq_len
        
        # Reshape to 2D for processing: [batch_size * seq_len, hidden_size]
        x_reshaped = x.view(-1, hidden_size)
        
        # Create output with same shape
        y = torch.empty_like(x_reshaped)
        
        # Set kernel parameters
        BLOCK_SIZE = min(1024, n_elements_per_row)
        num_rows = batch_size * seq_len
        
        # Launch kernel
        grid = (num_rows,)
        layernorm_kernel[grid](
            x_ptr=x_reshaped,
            weight_ptr=weight,
            bias_ptr=bias,
            y_ptr=y,
            n_elements=n_elements_per_row,
            hidden_size=hidden_size,
            eps=1e-05,
            BLOCK_SIZE=BLOCK_SIZE
        )
        
        # Reshape back to original 3D shape
        return y.view(batch_size, seq_len, hidden_size)
    
    # Handle other dimensions as fallback
    else:
        return torch.nn.functional.layer_norm(x, normalized_shape, weight, bias, 1e-05)

def replacement_func():
    return optimized_layernorm