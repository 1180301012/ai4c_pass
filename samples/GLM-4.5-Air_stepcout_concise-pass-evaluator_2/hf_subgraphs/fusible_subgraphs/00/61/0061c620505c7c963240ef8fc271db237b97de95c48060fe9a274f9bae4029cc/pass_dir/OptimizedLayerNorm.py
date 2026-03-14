import torch
import triton
import triton.language as tl

def pattern(in_0, in_1, in_2):
    """Match layer_norm pattern exactly as in model.py"""
    tmp_0 = in_0
    tmp_1 = in_1
    tmp_2 = torch.nn.functional.layer_norm(in_2, (768,), tmp_1, tmp_0, 1e-05)
    return tmp_2

def replacement_args(in_0, in_1, in_2):
    return (in_0, in_1, in_2)

@triton.jit
def layer_norm_kernel(
    x_ptr,
    weight_ptr,
    bias_ptr,
    y_ptr,
    n_cols,
    rows,
    eps: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """High-performance layer normalization kernel"""
    # Each program computes one row of the output
    row_idx = tl.program_id(0)
    col_offset = tl.arange(0, BLOCK_SIZE)
    col_mask = col_offset < n_cols

    # Load bias and weight vectors
    bias = tl.load(bias_ptr + col_offset, mask=col_mask, other=0.0)
    weight = tl.load(weight_ptr + col_offset, mask=col_mask, other=1.0)
    
    # Compute mean for this row
    mean = tl.zeros([1], dtype=tl.float32)
    for col in range(0, n_cols, BLOCK_SIZE):
        cols = col + col_offset
        mask = cols < n_cols
        x = tl.load(x_ptr + row_idx * n_cols + cols, mask=mask, other=0.0)
        mean += tl.sum(x, axis=0)
    
    # Broadcast mean across BLOCK_SIZE and normalize
    mean = mean / n_cols
    
    # Compute variance
    variance = tl.zeros([1], dtype=tl.float32)
    for col in range(0, n_cols, BLOCK_SIZE):
        cols = col + col_offset
        mask = cols < n_cols
        x = tl.load(x_ptr + row_idx * n_cols + cols, mask=mask, other=0.0)
        x_centered = x - mean
        variance += tl.sum(x_centered * x_centered, axis=0)
    
    variance = variance / n_cols + eps
    
    # Compute output for current block
    x = tl.load(x_ptr + row_idx * n_cols + col_offset, mask=col_mask, other=0.0)
    std = tl.sqrt(variance)
    y = (x - mean) / std * weight + bias
    
    # Store output
    tl.store(y_ptr + row_idx * n_cols + col_offset, y, mask=col_mask)

@torch.fx.wrap
def optimized_layer_norm(input, weight, bias):
    """Optimized layer norm implementation"""
    # Reshape input from [batch, seq_len, hidden] to [batch*seq_len, hidden]
    original_shape = input.shape
    n_rows = original_shape[0] * original_shape[1]
    n_cols = original_shape[2]
    
    # Flatten batch and seq_len dimensions
    input_flat = input.reshape(n_rows, n_cols)
    
    # Create output
    output_flat = torch.empty_like(input_flat)
    
    # Launch layer norm kernel
    BLOCK_SIZE = 256
    grid = (n_rows,)
    layer_norm_kernel[grid](
        input_flat,
        weight,
        bias,
        output_flat,
        n_cols,
        n_rows,
        1e-5,
        BLOCK_SIZE
    )
    
    # Reshape back to original dimensions
    return output_flat.reshape(original_shape)

def replacement_func():
    return optimized_layer_norm