import torch
import triton
import triton.language as tl
import math

def pattern(input, normalized_shape, weight, bias, eps):
    return torch.nn.functional.layer_norm(input, normalized_shape, weight, bias, eps)

def replacement_args(input, normalized_shape, weight, bias, eps):
    return (input, normalized_shape, weight, bias, eps)

@triton.jit
def layernorm_kernel(
    input_ptr, weight_ptr, bias_ptr, out_ptr,
    n_cols, n_rows,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr,
):
    # Each program processes a BLOCK_SIZE_M x BLOCK_SIZE_N tile
    pid_m = tl.program_id(0)  # row id
    pid_n = tl.program_id(1)  # column id
    
    # Create pointers to the start of the tile
    input_ptr = input_ptr + pid_m * n_cols
    weight_ptr = weight_ptr + pid_n
    bias_ptr = bias_ptr + pid_n
    out_ptr = out_ptr + pid_m * n_cols + pid_n
    
    # Load weight and bias for this column
    if weight_ptr is not None:
        weight_val = tl.load(weight_ptr)
    else:
        weight_val = 1.0
        
    if bias_ptr is not None:
        bias_val = tl.load(bias_ptr)
    else:
        bias_val = 0.0
    
    # Load tile data into SRAM
    row_start = tl.program_id(0) * BLOCK_SIZE_M
    rows_per_program = min(BLOCK_SIZE_M, n_rows - row_start)
    
    # Initialize accumulators for mean and variance
    mean = 0.0
    var = 0.0
    
    # Compute mean and variance for this row
    for m in range(rows_per_program):
        offset = (row_start + m) * n_cols + pid_n
        if offset < n_rows * n_cols:
            val = tl.load(input_ptr + m * n_cols)
            mean += val
            var += val * val
    
    # Reduce across the thread block
    block_size = tl.num_programs(0) * block_size // tl.num_programs(1)
    mean = tl.sum(mean) / (rows_per_program)
    var = tl.sum(var) / (rows_per_program) - mean * mean
    
    # Normalize the tile
    for m in range(rows_per_program):
        offset = (row_start + m) * n_cols
        if offset < n_rows * n_cols:
            val = tl.load(input_ptr + m * n_cols)
            normalized = (val - mean) / tl.sqrt(var + 1e-5)
            result = normalized * weight_val + bias_val
            tl.store(out_ptr + m * n_cols, result)

# Optimized LayerNorm kernel with better memory access patterns
@triton.jit
def optimized_layernorm_kernel(
    X_ptr, W_ptr, B_ptr, Y_ptr,
    n_rows, n_cols,
    eps: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr,
):
    # Each program processes one row
    row_idx = tl.program_id(0)
    
    # Compute row offset
    row_ptr = X_ptr + row_idx * n_cols
    
    # Initialize mean and variance accumulators
    row_mean = 0.0
    row_var = 0.0
    
    # Process columns in chunks to better utilize registers/SRAM
    for col_start in range(0, n_cols, BLOCK_SIZE_N):
        col_end = min(col_start + BLOCK_SIZE_N, n_cols)
        block_mask = col_start + tl.arange(0, BLOCK_SIZE_N) < n_cols
        
        # Load input block
        X_block = tl.load(row_ptr + col_start + tl.arange(0, BLOCK_SIZE_N), mask=block_mask, other=0.0)
        
        # Update mean and variance accumulators
        row_mean += tl.sum(X_block)
        row_var += tl.sum(X_block * X_block)
    
    # Compute final mean and variance for the row
    row_mean = row_mean / n_cols
    row_var = row_var / n_cols - row_mean * row_var
    
    # Apply normalization
    sqrt_var = tl.sqrt(row_var + eps)
    for col_start in range(0, n_cols, BLOCK_SIZE_N):
        col_end = min(col_start + BLOCK_SIZE_N, n_cols)
        block_mask = col_start + tl.arange(0, BLOCK_SIZE_N) < n_cols
        
        # Load input block
        X_block = tl.load(row_ptr + col_start + tl.arange(0, BLOCK_SIZE_N), mask=block_mask, other=0.0)
        
        # Apply normalization and parameters
        X_normalized = (X_block - row_mean) / sqrt_var
        
        # Load weight and bias if they exist
        if W_ptr is not None:
            W_block = tl.load(W_ptr + col_start + tl.arange(0, BLOCK_SIZE_N), mask=block_mask, other=1.0)
        else:
            W_block = 1.0
            
        if B_ptr is not None:
            B_block = tl.load(B_ptr + col_start + tl.arange(0, BLOCK_SIZE_N), mask=block_mask, other=0.0)
        else:
            B_block = 0.0
        
        # Compute output
        Y_block = X_normalized * W_block + B_block
        
        # Store result
        Y_row_ptr = Y_ptr + row_idx * n_cols
        tl.store(Y_row_ptr + col_start + tl.arange(0, BLOCK_SIZE_N), Y_block, mask=block_mask)

@torch.fx.wrap
def optimized_layer_norm(input, normalized_shape, weight, bias, eps):
    # Handle both 2D and 3D inputs
    if input.dim() == 3:
        # Input shape: [1, seq_len, hidden_size] -> process as [seq_len, hidden_size]
        input = input.squeeze(0)  # Remove batch dimension
    
    n_rows, n_cols = input.shape
    
    # Create output tensor
    output = torch.empty_like(input)
    
    # Set block sizes based on tensor dimensions
    BLOCK_SIZE_M = 64  # Process multiple rows per program
    BLOCK_SIZE_N = 256  # Process multiple columns per program
    
    # Calculate grid size
    grid_m = (n_rows + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M
    grid_n = (n_cols + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N
    
    # Launch the optimized kernel
    optimized_layernorm_kernel[(grid_m, grid_n)](
        input,
        weight,
        bias,
        output,
        n_rows,
        n_cols,
        eps,
        BLOCK_SIZE_M,
        BLOCK_SIZE_N,
    )
    
    # Restore original dimensionality
    if input.dim() == 3:
        output = output.unsqueeze(0)  # Add back batch dimension
    
    return output

def replacement_func():
    return optimized_layer_norm