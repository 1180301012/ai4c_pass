import torch
import triton
import triton.language as tl

def pattern(tmp_7, weight, bias):
    # tmp_7: input tensor after multiplication operations
    # weight: layer norm weight [320]
    # bias: layer norm bias [320]
    tmp_8 = torch.nn.functional.layer_norm(tmp_7, (320,), weight, bias, 1e-05)
    return tmp_8

def replacement_args(tmp_7, weight, bias):
    return (tmp_7, weight, bias)

@triton.jit
def layernorm_kernel(
    x_ptr, weight_ptr, bias_ptr, out_ptr,
    M, N,
    eps: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    # Initialize program ID
    pid = tl.program_id(0)
    
    # Compute row range
    row_start = pid * BLOCK_SIZE_M
    row_end = min((pid + 1) * BLOCK_SIZE_M, M)
    
    # Create empty rows for mean and variance
    mean = tl.zeros((BLOCK_SIZE_M,), dtype=tl.float32)
    var = tl.zeros((BLOCK_SIZE_M,), dtype=tl.float32)
    
    # Reduction pass: compute mean and variance
    for i in range(row_start, row_end):
        # Load current row
        row_ptrs = x_ptr + i * N
        row = tl.load(row_ptrs, mask=tl.arange(0, N) < N, other=0.0)
        
        # Compute sum of squares
        row_sum = tl.sum(row)
        row_sum_sq = tl.sum(row * row)
        
        # Accumulate to mean and var
        mean[pid] = row_sum
        var[pid] = row_sum_sq
    
    # Synchronize within threads and compute final mean and variance
    if BLOCK_SIZE_M > 1:
        mean = tl.sum(mean, axis=0)
        var = tl.sum(var, axis=0)
        M_total = M * BLOCK_SIZE_M
    else:
        M_total = M
    
    # Final mean and variance
    final_mean = mean / N
    final_var = (var / N) - (final_mean * final_mean)
    final_var = tl.maximum(final_var, tl.constexpr(0.0))
    final_std = tl.sqrt(final_var + eps)
    
    # Normalization pass: apply layer norm
    for i in range(row_start, row_end):
        # Load current row and parameters
        row_ptrs = x_ptr + i * N
        row = tl.load(row_ptrs, mask=tl.arange(0, N) < N, other=0.0)
        
        # Load weight and bias
        w = tl.load(weight_ptr, mask=tl.arange(0, N) < N, other=0.0)
        b = tl.load(bias_ptr, mask=tl.arange(0, N) < N, other=0.0)
        
        # Apply layer normalization
        normalized = (row - final_mean) / final_std
        out = normalized * w + b
        
        # Store result
        out_ptrs = out_ptr + i * N
        tl.store(out_ptrs, out, mask=tl.arange(0, N) < N)

@torch.fx.wrap
def optimized_layer_norm(x, weight, bias):
    # Get input dimensions
    M, N = x.shape  # M = batch_size*seq_len, N = hidden_dim (320)
    
    # Set block sizes
    BLOCK_SIZE_M = 32  # Number of rows per block
    BLOCK_SIZE_N = 32  # Number of columns per block (since N=320, this works well)
    
    # Calculate number of programs
    num_programs = (M + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M
    
    # Create output tensor
    out = torch.empty_like(x, dtype=torch.float32)
    
    # Launch kernel
    layernorm_kernel[(num_programs,)](
        x, weight, bias, out,
        M, N,
        1e-05,
        BLOCK_SIZE_M,
        BLOCK_SIZE_N,
    )
    
    return out

def replacement_func():
    return optimized_layer_norm