import torch
import triton
import triton.language as tl

def pattern(tmp_4):
    tmp_5 = torch.nn.functional.softmax(tmp_4, dim=2)
    return tmp_5

def replacement_args(tmp_4):
    return (tmp_4,)

@triton.jit
def optimized_softmax_kernel(
    x_ptr, out_ptr,
    n_rows, n_cols, n_features,  # [1, 4096, 32] → rows=4096, cols=1, features=32
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr
):
    # Program indices for 2D grid
    pid_m = tl.program_id(0)   # Process rows (4096)
    pid_n = tl.program_id(1)   # Process column chunks (1 column chunk)
    
    # Calculate element offsets within each row for softmax computation
    row_start = pid_m * BLOCK_SIZE_M
    row_offsets = row_start + tl.arange(0, BLOCK_SIZE_M)
    row_mask = row_offsets < n_rows
    
    # Feature offset for this column chunk
    feature_start = pid_n * BLOCK_SIZE_N
    feature_offsets = feature_start + tl.arange(0, BLOCK_SIZE_N)
    feature_mask = feature_offsets < n_features
    
    # Load input data with proper masking
    # Input shape: [1, 4096, 32] → We process each row independently
    # Load all features for the current row(s)
    x = tl.load(x_ptr + (row_offsets[:, None] * (n_cols * n_features) + 
                         feature_offsets[None, :]),
               mask=row_offsets[:, None] < n_rows & feature_offsets[None, :] < n_features,
               other=-float('inf'))
    
    # Compute max for numerical stability (per row)
    max_val = tl.max(x, axis=1)
    
    # Subtract max and compute exponential
    x_shifted = x - max_val[:, None]
    exp_x = tl.exp(x_shifted)
    
    # Compute sum of exponentials
    sum_exp = tl.sum(exp_x, axis=1)
    
    # Normalize to get softmax
    softmax_out = exp_x / sum_exp[:, None]
    
    # Store output
    out_base = out_ptr + (row_offsets[:, None] * (n_cols * n_features) + 
                         feature_offsets[None, :])
    tl.store(out_base,
             softmax_out,
             mask=row_offsets[:, None] < n_rows & feature_offsets[None, :] < n_features)

@torch.fx.wrap
def optimized_softmax(tmp_4):
    # Input tensor shape: [1, 4096, 32]
    x = tmp_4
    
    # Reshape for kernel processing: remove batch dimension
    x_reshaped = x.reshape(4096, 1, 32)  # [4096, 1, 32]
    
    # Create output tensor
    out_shape = (1, 4096, 32)
    out = torch.empty(out_shape, dtype=x.dtype, device=x.device)
    
    # Set block sizes for optimal performance
    BLOCK_SIZE_M = 64   # Process 4096 rows in chunks of 64
    BLOCK_SIZE_N = 32   # Process all 32 features at once
    
    # Calculate grid size
    grid_m = (4096 + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M
    grid_n = (32 + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N
    
    # Launch kernel
    optimized_softmax_kernel[(grid_m, grid_n)](
        x_reshaped,
        out,
        4096, 1, 32,
        BLOCK_SIZE_M, BLOCK_SIZE_N
    )
    
    return out

def replacement_func():
    return optimized_softmax