import torch
import triton
import triton.language as tl

def pattern(x, normalized_shape, weight, bias, eps):
    """Pattern: single layer_norm"""
    result = torch.nn.functional.layer_norm(x, normalized_shape, weight, bias, eps)
    return result

def replacement_args(x, normalized_shape, weight, bias, eps):
    return (x, normalized_shape, weight, bias, eps)

@triton.jit
def layer_norm_kernel(
    input_ptr,
    weight_ptr,
    bias_ptr,
    output_ptr,
    n_rows,
    n_cols,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Optimized LayerNorm kernel
    Computes: output = (input - mean) / sqrt(variance + eps) * weight + bias
    Each row is processed by one block
    """
    row_idx = tl.program_id(0)
    
    if row_idx >= n_rows:
        return
    
    # Compute row offset
    row_start = row_idx * n_cols
    
    # Process the row in one pass if possible, or multiple passes
    col_offsets = tl.arange(0, BLOCK_SIZE)
    
    # Pass 1: Compute mean
    mean_acc = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for block_start in range(0, n_cols, BLOCK_SIZE):
        offsets = block_start + col_offsets
        mask = offsets < n_cols
        data = tl.load(input_ptr + row_start + offsets, mask=mask, other=0.0)
        mean_acc += data
    
    mean = tl.sum(mean_acc) / n_cols
    
    # Pass 2: Compute variance
    var_acc = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for block_start in range(0, n_cols, BLOCK_SIZE):
        offsets = block_start + col_offsets
        mask = offsets < n_cols
        data = tl.load(input_ptr + row_start + offsets, mask=mask, other=0.0)
        centered = data - mean
        var_acc += centered * centered
    
    variance = tl.sum(var_acc) / n_cols
    rstd = 1.0 / tl.sqrt(variance + eps)
    
    # Pass 3: Normalize and apply affine transformation
    for block_start in range(0, n_cols, BLOCK_SIZE):
        offsets = block_start + col_offsets
        mask = offsets < n_cols
        
        data = tl.load(input_ptr + row_start + offsets, mask=mask, other=0.0)
        weight = tl.load(weight_ptr + offsets, mask=mask, other=1.0)
        bias_val = tl.load(bias_ptr + offsets, mask=mask, other=0.0)
        
        normalized = (data - mean) * rstd
        output = normalized * weight + bias_val
        
        tl.store(output_ptr + row_start + offsets, output, mask=mask)

@torch.fx.wrap
def optimized_layer_norm(x, normalized_shape, weight, bias, eps):
    """
    Optimized LayerNorm implementation
    x: input tensor of any shape
    normalized_shape: shape to normalize over (last dimensions)
    weight: learnable weight
    bias: learnable bias
    eps: epsilon for numerical stability
    """
    # Flatten to 2D for processing
    original_shape = x.shape
    n_cols = 1
    for dim in normalized_shape:
        n_cols *= dim
    
    # Reshape to [n_rows, n_cols]
    x_2d = x.reshape(-1, n_cols)
    n_rows = x_2d.shape[0]
    
    output = torch.empty_like(x_2d)
    
    BLOCK_SIZE = 1024
    grid = (n_rows,)
    
    layer_norm_kernel[grid](
        x_2d,
        weight,
        bias,
        output,
        n_rows,
        n_cols,
        eps,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    # Reshape back to original shape
    return output.reshape(original_shape)

def replacement_func():
    return optimized_layer_norm