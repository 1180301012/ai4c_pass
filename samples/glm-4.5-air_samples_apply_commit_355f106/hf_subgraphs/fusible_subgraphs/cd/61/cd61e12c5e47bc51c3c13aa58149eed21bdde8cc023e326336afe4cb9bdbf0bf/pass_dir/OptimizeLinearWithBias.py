import torch
import triton
import triton.language as tl

def pattern(x, weight, bias):
    # Linear operation: y = x @ weight.T + bias
    result = torch.nn.functional.linear(x, weight, bias)
    return result

def replacement_args(x, weight, bias):
    return (x, weight, bias)

@triton.jit
def linear_bias_kernel(
    x_ptr, weight_ptr, bias_ptr, out_ptr,
    n_rows, n_cols, n_cols_out,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
):
    # Each program computes a tile of the output matrix
    pid_m = tl.program_id(0)  # block row
    pid_n = tl.program_id(1)  # block column
    
    # Create offsets for the output tile
    offset_m = pid_m * BLOCK_SIZE_M
    offset_n = pid_n * BLOCK_SIZE_N
    
    # Initialize accumulator for the entire tile
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    # Create offsets within the tile
    offsets_m = offset_m + tl.arange(0, BLOCK_SIZE_M)
    offsets_n = offset_n + tl.arange(0, BLOCK_SIZE_N)
    mask = offsets_m[:, None] < n_rows and offsets_n[None, :] < n_cols_out
    
    # Load bias vector
    bias = tl.load(bias_ptr + offsets_n, mask=offsets_n < n_cols_out, other=0.0)
    
    # Vectorized matrix multiplication with tiling
    for k in range(0, tl.cdiv(n_cols, BLOCK_SIZE_K)):
        # Load weight tile for current k block (W has shape [n_cols_out, n_cols])
        offsets_k = k * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
        weights_k = tl.load(
            weight_ptr + offsets_n[:, None] * n_cols + offsets_k[None, :],
            mask=(offsets_n[:, None] < n_cols_out) & (offsets_k[None, :] < n_cols), 
            other=0.0
        )
        
        # Load x tile for current k block  
        x_k = tl.load(
            x_ptr + offsets_m[:, None] * n_cols + offsets_k[None, :],
            mask=(offsets_m[:, None] < n_rows) & (offsets_k[None, :] < n_cols), 
            other=0.0
        )
        
        # Compute matrix multiplication: result = x_k @ weights_k.T
        # x_k: [BLOCK_SIZE_M, BLOCK_SIZE_K], weights_k: [BLOCK_SIZE_N, BLOCK_SIZE_K]
        # We need x_k @ weights_k.T to give [BLOCK_SIZE_M, BLOCK_SIZE_N]
        acc += tl.dot(x_k, weights_k, out_dtype=tl.float32)
    
    # Add bias to the entire tile
    acc += bias[None, :]
    
    # Store the result tile
    out_offsets = offsets_m[:, None] * n_cols_out + offsets_n[None, :]
    tl.store(out_ptr + out_offsets, acc, mask=mask)

@torch.fx.wrap
def optimized_linear_bias(x, weight, bias):
    # Move weight and bias to GPU if they're on CPU
    if weight.device.type == 'cpu':
        weight = weight.to('cuda:0')
    if bias.device.type == 'cpu':
        bias = bias.to('cuda:0')
    
    n_rows, n_cols = x.shape
    n_cols_out = bias.shape[0]
    
    out = torch.empty((n_rows, n_cols_out), dtype=x.dtype, device=x.device)
    
    # Use optimal block sizes for [1000,128] x [128,128] -> [1000,128]
    BLOCK_SIZE_M = 32
    BLOCK_SIZE_N = 32
    BLOCK_SIZE_K = 32
    
    # Calculate grid dimensions
    num_blocks_m = triton.cdiv(n_rows, BLOCK_SIZE_M)
    num_blocks_n = triton.cdiv(n_cols_out, BLOCK_SIZE_N)
    
    grid = (num_blocks_m, num_blocks_n)
    
    linear_bias_kernel[grid](
        x, weight, bias, out,
        n_rows, n_cols, n_cols_out,
        BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K,
    )
    
    return out

def replacement_func():
    return optimized_linear_bias