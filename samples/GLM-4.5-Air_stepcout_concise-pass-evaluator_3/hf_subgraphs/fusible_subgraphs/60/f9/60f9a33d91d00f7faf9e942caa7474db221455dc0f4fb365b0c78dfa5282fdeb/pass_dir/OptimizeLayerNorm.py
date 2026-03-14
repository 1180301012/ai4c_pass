import torch
import triton
import triton.language as tl

def pattern(x, normalized_shape, weight, bias, eps):
    """Match torch.nn.functional.layer_norm operation"""
    return torch.nn.functional.layer_norm(x, normalized_shape, weight, bias, eps)

def replacement_args(x, normalized_shape, weight, bias, eps):
    return (x, normalized_shape, weight, bias, eps)

@triton.jit
def layer_norm_kernel(
    input_ptr,
    weight_ptr,
    bias_ptr,
    output_ptr,
    mean_ptr,
    rstd_ptr,
    n_element,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Triton kernel for layer normalization
    
    Args:
        input_ptr: pointer to input tensor [N, D]
        weight_ptr: pointer to weight [D] 
        bias_ptr: pointer to bias [D]
        output_ptr: pointer to output [N, D]
        mean_ptr: pointer to mean [N]
        rstd_ptr: pointer to 1/std [N]
        n_element: total number of elements in input
        eps: epsilon value for numerical stability
        BLOCK_SIZE: block size for the kernel
    """
    pid = tl.program_id(0)
    
    # Each program handles one batch element (row)
    row_idx = pid
    
    # Load weights and bias for this program
    weight = tl.load(weight_ptr + tl.arange(0, BLOCK_SIZE), mask=tl.arange(0, BLOCK_SIZE) < BLOCK_SIZE, other=1.0)
    bias = tl.load(bias_ptr + tl.arange(0, BLOCK_SIZE), mask=tl.arange(0, BLOCK_SIZE) < BLOCK_SIZE, other=0.0)
    
    # Calculate mean for this row
    row_start = row_idx * BLOCK_SIZE
    row_end = min(row_start + BLOCK_SIZE, n_element)
    row_block_size = row_end - row_start
    
    if row_block_size > 0:
        # Load input elements for this row
        input_vals = tl.load(input_ptr + row_start + tl.arange(0, row_block_size), mask=tl.arange(0, row_block_size) < row_block_size, other=0.0)
        
        # Calculate mean
        row_mean = tl.sum(input_vals) / row_block_size
        
        # Calculate variance
        diff = input_vals - row_mean
        row_var = tl.sum(diff * diff) / row_block_size
        
        # Calculate 1/std
        rstd = 1.0 / tl.sqrt(row_var + eps)
        
        # Store mean and rstd for this row
        if mean_ptr is not None:
            tl.store(mean_ptr + row_idx, row_mean)
        if rstd_ptr is not None:
            tl.store(rstd_ptr + row_idx, rstd)
        
        # Normalize and apply weight/bias
        output_vals = (input_vals - row_mean) * rstd * weight[:row_block_size] + bias[:row_block_size]
        
        # Store output
        tl.store(output_ptr + row_start + tl.arange(0, row_block_size), output_vals, mask=tl.arange(0, row_block_size) < row_block_size)

@triton.jit
def optimized_layer_norm_kernel(
    x_ptr,
    weight_ptr,
    bias_ptr,
    out_ptr,
    n_rows,
    n_cols,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Optimized Triton kernel for layer normalization with better memory access patterns
    
    Args:
        x_ptr: pointer to input tensor [n_rows, n_cols]
        weight_ptr: pointer to weight [n_cols]
        bias_ptr: pointer to bias [n_cols] 
        out_ptr: pointer to output [n_rows, n_cols]
        n_rows: number of rows in input
        n_cols: number of columns (features)
        eps: epsilon value
        BLOCK_SIZE: number of columns processed per program
    """
    pid = tl.program_id(0)
    
    # Process BLOCK_SIZE columns per program
    col_offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    col_mask = col_offsets < n_cols
    
    # Load weight and bias for the columns this program handles
    weight = tl.load(weight_ptr + col_offsets, mask=col_mask, other=1.0)
    bias = tl.load(bias_ptr + col_offsets, mask=col_mask, other=0.0)
    
    # Process each row
    for row_idx in range(n_rows):
        row_offset = row_idx * n_cols
        
        # Load input elements for this row and column block
        input_vals = tl.load(x_ptr + row_offset + col_offsets, mask=col_mask, other=0.0)
        
        # Calculate mean for this row
        row_sum = tl.sum(input_vals)
        row_mean = row_sum / tl.sum(col_mask)
        
        # Calculate variance (Welford's algorithm for better numerical stability)
        diff = input_vals - row_mean
        row_var = tl.sum(diff * diff) / tl.sum(col_mask)
        
        # Calculate standard deviation
        rstd = tl.math.rsqrt(row_var + eps)
        
        # Apply normalization, weight and bias
        output_vals = (input_vals - row_mean) * rstd * weight + bias
        
        # Store results
        tl.store(out_ptr + row_offset + col_offsets, output_vals, mask=col_mask)

@torch.fx.wrap
def optimized_layer_norm(x, normalized_shape, weight, bias, eps=1e-05):
    """
    Optimized layer normalization using Triton
    
    Args:
        x: input tensor [batch, seq_len, features]
        normalized_shape: shape for normalization (features,)
        weight: weight tensor [features]
        bias: bias tensor [features] 
        eps: epsilon value
    
    Returns:
        normalized tensor
    """
    batch, seq_len, features = x.shape
    total_elements = batch * seq_len
    
    # Reshape to 2D tensor for easier processing [batch*seq_len, features]
    x_2d = x.reshape(-1, features)
    out_2d = torch.empty_like(x_2d)
    
    # Autotune block size based on feature dimension
    if features <= 256:
        block_size = 64
    elif features <= 512:
        block_size = 128  
    else:
        block_size = 256
    
    # Calculate number of programs needed
    num_cols = features
    num_programs = (num_cols + block_size - 1) // block_size
    num_rows = total_elements
    
    # Launch kernel
    grid = (num_programs, num_rows)
    
    # Check if we need to adjust grid size
    if num_rows > 65535:  # Triton limit for program_id dimension
        # Fall back to original implementation for very large inputs
        return torch.nn.functional.layer_norm(x, normalized_shape, weight, bias, eps)
    
    optimized_layer_norm_kernel[grid](
        x_2d,
        weight,
        bias,
        out_2d,
        num_rows,
        num_cols,
        eps,
        BLOCK_SIZE=block_size,
    )
    
    # Reshape back to original format
    return out_2d.reshape(batch, seq_len, features)

def replacement_func():
    return optimized_layer_norm