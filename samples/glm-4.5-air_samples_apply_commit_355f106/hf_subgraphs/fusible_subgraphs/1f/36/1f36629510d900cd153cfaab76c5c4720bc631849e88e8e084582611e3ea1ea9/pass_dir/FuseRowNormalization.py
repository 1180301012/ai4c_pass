import torch
import triton
import triton.language as tl

# Pattern matching function - matches the normalization sequence
def pattern(x):
    tmp_0 = x.sum(dim=-1)
    tmp_1 = tmp_0.unsqueeze(-1)
    out = x / tmp_1
    return out

# Argument extraction function
def replacement_args(x):
    return (x,)

# Optimized kernel for row-wise normalization
@triton.jit
def row_normalization_kernel(
    x_ptr,
    out_ptr,
    n_batch,
    n_heads,
    n_seq,
    n_features,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    # Each program handles one row of the matrix
    batch_idx = tl.program_id(0)
    head_idx = tl.program_id(1)
    row_idx = tl.program_id(2)
    
    # Calculate pointer offsets
    x_batch_offset = batch_idx * n_heads * n_seq * n_features
    x_head_offset = head_idx * n_seq * n_features
    x_row_offset = row_idx * n_features
    
    # Load the entire row
    row_start_ptr = x_ptr + x_batch_offset + x_head_offset + x_row_offset
    row = tl.load(row_start_ptr + tl.arange(0, n_features), mask=tl.arange(0, n_features) < n_features)
    
    # Compute sum of the row
    row_sum = tl.sum(row, 0)
    
    # Normalize the row by dividing by sum
    normalized_row = row / (row_sum + tl.where(row_sum == 0, tl.float32(1e-8), tl.float32(0)))
    
    # Store the result
    out_batch_offset = batch_idx * n_heads * n_seq * n_features
    out_head_offset = head_idx * n_seq * n_features
    out_row_offset = row_idx * n_features
    
    out_start_ptr = out_ptr + out_batch_offset + out_head_offset + out_row_offset
    tl.store(out_start_ptr + tl.arange(0, n_features), normalized_row, mask=tl.arange(0, n_features) < n_features)

@torch.fx.wrap
def triton_row_normalization(x):
    batch, heads, seq, features = x.shape
    
    # Define launch grid
    grid = (batch, heads, seq)
    
    # Output tensor
    out = torch.empty_like(x)
    
    # Launch kernel
    row_normalization_kernel[grid](
        x_ptr=x,
        out_ptr=out,
        n_batch=batch,
        n_heads=heads,
        n_seq=seq,
        n_features=features,
        BLOCK_SIZE_M=128,
        BLOCK_SIZE_N=128,
    )
    
    return out

def replacement_func():
    return triton_row_normalization