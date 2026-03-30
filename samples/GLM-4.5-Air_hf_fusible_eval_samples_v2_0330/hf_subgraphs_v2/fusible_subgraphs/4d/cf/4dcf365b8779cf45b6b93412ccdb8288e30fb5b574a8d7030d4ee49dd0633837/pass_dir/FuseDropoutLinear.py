import torch
import triton
import triton.language as tl

# Pattern 1: Dropout + Linear fusion
def pattern(in_2, in_1, in_0):
    tmp_3 = torch.nn.functional.dropout(in_2, 0.1, False, False)
    linear = torch.nn.functional.linear(tmp_3, in_1, in_0)
    return linear

def replacement_args(in_2, in_1, in_0):
    return (in_2, in_1, in_0)

# Optimized kernel that fuses dropout and linear operation
@triton.jit
def dropout_linear_kernel(
    x_ptr,
    weight_ptr,
    bias_ptr,
    out_ptr,
    stride_w0,
    stride_w1,
    n_rows,
    n_cols,
    n_features,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    # Dropout rate
    p = 0.1
    
    # Program ID
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    # Range of rows this program should compute (output rows)
    row_start = pid_m * BLOCK_SIZE_M
    row_end = min(row_start + BLOCK_SIZE_M, n_rows)
    rows = row_start + tl.arange(0, BLOCK_SIZE_M)
    mask_m = rows < n_rows
    
    # Range of columns this program should compute (output columns)
    col_start = pid_n * BLOCK_SIZE_N
    col_end = min(col_start + BLOCK_SIZE_N, n_cols)
    cols = col_start + tl.arange(0, BLOCK_SIZE_N)
    mask_n = cols < n_cols
    
    # Initialize accumulator for this block
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    # Matrix multiplication loop
    for k in range(0, n_features, BLOCK_SIZE_K):
        # Load input block (features dimension)
        k_mask = k + tl.arange(0, BLOCK_SIZE_K) < n_features
        x_block = tl.load(
            x_ptr + (k + tl.arange(0, BLOCK_SIZE_K))[:, None],
            mask=k_mask[:, None],
            other=0.0
        )
        
        # Apply dropout scaling
        x_scaled = x_block * (1.0 / (1.0 - p))
        
        # Load weight block  
        weight_ptrs = weight_ptr + (k + tl.arange(0, BLOCK_SIZE_K))[:, None] * stride_w0 + cols[None, :]
        weight_block = tl.load(
            weight_ptrs,
            mask=k_mask[:, None] & mask_n[None, :],
            other=0.0
        )
        
        # Vectorized outer product approach
        x_vec = x_scaled.to(tl.float32).reshape([BLOCK_SIZE_K])  # [K]
        weight_vec = weight_block.to(tl.float32).reshape([BLOCK_SIZE_K, BLOCK_SIZE_N])  # [K, N]
        
        # Compute outer product and sum along K dimension to get [M, N]
        outer_product = x_vec[None, :] * weight_vec  # [1, K] * [K, N] -> [1, N]
        accumulator += outer_product
    
    # Load bias and add
    bias = tl.load(bias_ptr + cols[None, :], mask=mask_n[None, :], other=0.0)
    accumulator += bias
    
    # Store result
    tl.store(
        out_ptr + rows[:, None] * n_cols + cols[None, :],
        accumulator,
        mask=mask_m[:, None] & mask_n[None, :]
    )

@torch.fx.wrap
def fused_dropout_linear(x, weight, bias):
    # Get input dimensions
    batch_size, seq_len, n_features = x.shape
    n_rows = weight.shape[0]  # 3072 for BigBird
    
    # Reshape input for linear operation: [batch_size*seq_len, n_features]
    x_flat = x.reshape(-1, n_features)
    n_cols = weight.shape[1]  # Should be n_features (768 for BigBird)
    
    # Output shape: [batch_size*seq_len, n_rows]
    out = torch.empty((x_flat.shape[0], n_rows), dtype=x.dtype, device=x.device)
    
    # Block sizes for GPU optimization (all dimensions >= 16 to meet Triton constraints)
    BLOCK_SIZE_M = 16
    BLOCK_SIZE_N = 16
    BLOCK_SIZE_K = 16
    
    # Grid calculation
    grid_m = (x_flat.shape[0] + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M
    grid_n = (n_rows + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N
    
    # Launch kernel
    dropout_linear_kernel[(grid_m, grid_n)](
        x_flat,
        weight,
        bias,
        out,
        weight.stride(0),
        weight.stride(1),
        x_flat.shape[0],
        n_rows,
        n_features,
        BLOCK_SIZE_M,
        BLOCK_SIZE_N,
        BLOCK_SIZE_K
    )
    
    # Reshape back to original batch shape
    return out.reshape(batch_size, seq_len, n_rows)

def replacement_func():
    return fused_dropout_linear