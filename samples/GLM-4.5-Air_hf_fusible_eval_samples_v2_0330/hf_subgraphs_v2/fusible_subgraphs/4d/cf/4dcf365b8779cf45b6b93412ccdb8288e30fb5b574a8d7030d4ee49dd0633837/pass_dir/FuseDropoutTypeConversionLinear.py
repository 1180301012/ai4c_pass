import torch
import triton
import triton.language as tl

# Pattern 2: Dropout + Type Conversion + Linear fusion
def pattern(in_2, in_1, in_0):
    tmp_2 = torch.nn.functional.dropout(in_2, p=0.0, training=False)
    to = tmp_2.to(in_1.dtype)  # Convert to weight dtype
    linear = torch.nn.functional.linear(to, in_1, in_0)
    return linear

def replacement_args(in_2, in_1, in_0):
    return (in_2, in_1, in_0)

# Optimized kernel that fuses dropout, type conversion, and linear operation
@triton.jit
def dropout_conversion_linear_kernel(
    x_ptr,
    weight_ptr,
    bias_ptr,
    out_ptr,
    stride_w0,
    stride_w1,
    output_dtype,
    n_rows,
    n_cols,
    n_features,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    # Dropout rate (0.0 means no dropout)
    p = 0.0
    
    # Program ID
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    # Range of rows this program should compute.
    row_start = pid_m * BLOCK_SIZE_M
    row_end = min(row_start + BLOCK_SIZE_M, n_rows)
    rows = row_start + tl.arange(0, BLOCK_SIZE_M)
    mask_m = rows < n_rows
    
    # Range of columns this program should compute.
    col_start = pid_n * BLOCK_SIZE_N
    col_end = min(col_start + BLOCK_SIZE_N, n_cols)
    cols = col_start + tl.arange(0, BLOCK_SIZE_N)
    mask_n = cols < n_cols
    
    # Compute block of output
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    for k in range(0, n_features, BLOCK_SIZE_K):
        # Load input block
        x_block = tl.load(
            x_ptr + k + tl.arange(0, BLOCK_SIZE_K)[:, None],
            mask=k + tl.arange(0, BLOCK_SIZE_K)[:, None] < n_features,
            other=0.0
        )
        
        # Since p=0, dropout is essentially no-op, but we keep the logic for future extension
        x_dropout = tl.where(x_block == 0.0, 0.0, x_block * (1.0 / (1.0 - p)))
        
        # Load weight block
        weight_block = tl.load(
            weight_ptr + k * stride_w0 + cols[None, :],
            mask=k + tl.arange(0, 1)[:, None] < n_features,
            other=0.0
        )
        
        # Accumulate matrix multiplication
        accumulator += x_dropout[:, None] * weight_block[None, :, :]
    
    # Load bias
    bias = tl.load(bias_ptr + cols, mask=mask_n, other=0.0)
    
    # Add bias
    accumulator += bias[None, :]
    
    # Convert to target dtype and store
    if output_dtype == 1:  # torch.float16
        out = accumulator.to(tl.float16)
    elif output_dtype == 2:  # torch.bfloat16
        out = accumulator.to(tl.bfloat16)
    else:  # torch.float32
        out = accumulator
    
    tl.store(
        out_ptr + rows[:, None] * n_cols + cols[None, :],
        out,
        mask=mask_m[:, None] & mask_n[None, :]
    )

@torch.fx.wrap
def fused_dropout_conversion_linear(x, weight, bias):
    # Get input dimensions
    n_rows = weight.shape[0]  # 128 for RECT_L
    n_features = weight.shape[1]  # 128 for RECT_L
    
    # Input is [128, 128], which is the feature dimension
    out_rows = x.shape[0]  # Should be 128
    out_cols = n_rows  # Should be 128
    
    # Output shape: [out_rows, out_cols]
    out = torch.empty((out_rows, out_cols), dtype=weight.dtype, device=x.device)
    
    # Block sizes for GPU optimization (optimized for 128x128 matrices)
    BLOCK_SIZE_M = 32
    BLOCK_SIZE_N = 32
    BLOCK_SIZE_K = 32
    
    # Determine dtype encoding for Triton
    if weight.dtype == torch.float16:
        dtype_code = 1
    elif weight.dtype == torch.bfloat16:
        dtype_code = 2
    else:
        dtype_code = 3  # float32
    
    # Grid calculation
    grid_m = (out_rows + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M
    grid_n = (out_cols + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N
    
    # Launch kernel
    dropout_conversion_linear_kernel[(grid_m, grid_n)](
        x,
        weight,
        bias,
        out,
        weight.stride(0),
        weight.stride(1),
        dtype_code,
        out_rows,
        out_cols,
        n_features,
        BLOCK_SIZE_M,
        BLOCK_SIZE_N,
        BLOCK_SIZE_K
    )
    
    return out

def replacement_func():
    return fused_dropout_conversion_linear