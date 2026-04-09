import torch
import triton
import triton.language as tl

def pattern(x, weight, bias):
    """
    Match the layer normalization operation that can be optimized with Triton
    tmp_13 = torch.nn.functional.layer_norm(tmp_12, (768,), in_5, in_4, 1e-06)
    x = tmp_12, weight = in_5, bias = in_4
    """
    tmp_13 = torch.nn.functional.layer_norm(x, (768,), weight, bias, 1e-06)
    return tmp_13

def replacement_args(x, weight, bias):
    return (x, weight, bias)

@triton.jit
def layer_norm_kernel(
    x_ptr,
    weight_ptr,
    bias_ptr,
    out_ptr,
    n_cols: tl.constexpr,
    eps: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
):
    """
    Triton kernel for layer normalization
    """
    # Block of columns that this program will process
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols
    
    # Allocate shared memory for batch statistics
    row_mean = tl.zeros([1], dtype=tl.float32)
    row_var = tl.zeros([1], dtype=tl.float32)
    
    # Load weight and bias (same for all rows in this group)
    weight = tl.load(weight_ptr + col_offsets, mask=mask, other=1.0).to(tl.float32)
    bias = tl.load(bias_ptr + col_offsets, mask=mask, other=0.0).to(tl.float32)
    
    # Get row and column pid
    pid = tl.program_id(0)
    n_rows = tl.program_id(1)  # We need 2D program ids for this
    
    # Iterate through all rows to compute mean and variance
    row_offset = pid * GROUP_SIZE
    for row_idx in range(row_offset, row_offset + GROUP_SIZE):
        # Load row data
        x_row = tl.load(x_ptr + row_idx * n_cols + col_offsets, mask=mask, other=0.0).to(tl.float32)
        
        # Compute mean
        current_mean = tl.sum(x_row) / tl.sum(col_offsets < n_cols)
        if row_idx == row_offset:
            row_mean = current_mean
        
        # Compute variance
        x_centered = x_row - current_mean
        current_var = tl.sum(x_centered * x_centered) / tl.sum(col_offsets < n_cols)
        if row_idx == row_offset:
            row_var = current_var
    
    # All threads in the block must see the same mean and variance
    tl.device_barrier()
    
    # Process each row in the group
    for row_idx in range(row_offset, row_offset + GROUP_SIZE):
        # Load row data
        x_row = tl.load(x_ptr + row_idx * n_cols + col_offsets, mask=mask, other=0.0).to(tl.float32)
        
        # Normalize
        denom = tl.rsqrt(row_var + eps)
        x_normalized = (x_row - row_mean) * denom
        
        # Apply weight and bias, convert back to original dtype
        x_out = x_normalized * weight + bias
        out_data = x_out.to(tl.float32)  # Assuming input is float16/bfloat16
        
        # Store output
        tl.store(out_ptr + row_idx * n_cols + col_offsets, out_data, mask=mask)

@torch.fx.wrap
def optimized_layer_norm(x, weight, bias):
    """
    Optimized layer normalization using Triton kernel
    """
    n_rows, n_cols = x.shape
    
    # Check if we need to reshape (input might be 3D, we take the last two dims)
    if x.dim() == 3:
        # Reshape to 2D for layer norm
        x_reshaped = x.reshape(-1, n_cols)
        weight = weight
        bias = bias
    else:
        x_reshaped = x
    
    batch_size, hidden_size = x_reshaped.shape
    
    # Create output tensor
    out = torch.empty_like(x_reshaped)
    
    # Choose block and group sizes
    BLOCK_SIZE = 256
    GROUP_SIZE = 32  # Number of rows to process together for statistics
    
    # Check if we can use the 2D kernel version
    if GROUP_SIZE * GROUP_SIZE < batch_size:
        # Use 2D grid
        grid = (
            (batch_size + GROUP_SIZE - 1) // GROUP_SIZE,
            GROUP_SIZE,
        )
        
        layer_norm_kernel[grid](
            x_reshaped,
            weight,
            bias,
            out,
            hidden_size,
            1e-06,
            BLOCK_SIZE=BLOCK_SIZE,
            GROUP_SIZE=GROUP_SIZE,
        )
    else:
        # If batch size is small, use simpler approach
        # This prevents grid size issues
        row_groups = (batch_size + GROUP_SIZE - 1) // GROUP_SIZE
        grid = (row_groups, GROUP_SIZE)
        
        layer_norm_kernel[grid](
            x_reshaped,
            weight,
            bias,
            out,
            hidden_size,
            1e-06,
            BLOCK_SIZE=BLOCK_SIZE,
            GROUP_SIZE=GROUP_SIZE,
        )
    
    # Reshape back if needed
    if x.dim() == 3:
        out = out.reshape(x.shape)
    
    return out

def replacement_func():
    return optimized_layer_norm