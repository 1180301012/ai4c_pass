import torch
import triton
import triton.language as tl

# Pattern matching for linear transformation
def linear_pattern(x, weight):
    # Linear = x @ weight.T (since weight is [out_features, in_features])
    result = x @ weight.T
    return result

# Argument extraction function
def replacement_args(x, weight):
    return (x, weight)

# Triton kernel for optimized linear transformation (matrix multiplication)
@triton.jit
def linear_kernel(
    x_ptr, 
    weight_ptr, 
    output_ptr,
    n_cols, 
    n_rows, 
    BLOCK_SIZE_M: tl.constexpr, 
    BLOCK_SIZE_N: tl.constexpr, 
    BLOCK_SIZE_K: tl.constexpr,
):
    # Program id and block size for 2D grid
    pid = tl.program_id(0)
    num_programs = tl.cdiv(n_rows, BLOCK_SIZE_M)
    pid_m = pid // num_programs
    pid_n = pid % num_programs
    
    # Compute block start addresses
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    
    # Create masks for boundaries
    mask_m = offs_m < n_rows
    mask_n = offs_n < n_cols
    
    # Allocate block shared memory in registers (Triton handles this automatically)
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    # Loop over K dimension
    for k in range(0, tl.cdiv(32, BLOCK_SIZE_K)):
        # Load x block [M, K]
        x_ptrs = x_ptr + (offs_m[:, None] * 32 + k * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)[None, :])
        x_block = tl.load(x_ptrs, mask=mask_m[:, None] & (tl.arange(0, BLOCK_SIZE_K)[None, :] < 32), other=0.0)
        x_block = x_block.to(tl.float32)
        
        # Load weight block [K, N]
        weight_ptrs = weight_ptr+ (tl.arange(0, BLOCK_SIZE_K)[:, None] * n_cols + offs_n[None, :])
        weight_block = tl.load(weight_ptrs.T, mask=(tl.arange(0, BLOCK_SIZE_K)[:, None] < 32) & mask_n[None, :], other=0.0)
        weight_block = weight_block.to(tl.float32)
        
        # Matrix multiplication
        accumulator += tl.dot(x_block, weight_block, out_precision=tl.float32)
    
    # Store result with proper data type conversion
    output_ptrs = output_ptr + (offs_m[:, None] * n_cols + offs_n[None, :])
    tl.store(output_ptrs, accumulator.to(tl.bfloat16), mask=mask_m[:, None] & mask_n[None, :])

@torch.fx.wrap
def optimized_linear(x, weight):
    # Get input dimensions
    n_rows, n_cols = x.shape  # x: [1000, 32], weight: [16, 32], output: [1000, 16]
    k_dim = 32  # This is the common dimension
    
    # Set block sizes based on typical optimal sizes for GPU
    BLOCK_SIZE_M = 64   # Rows per block
    BLOCK_SIZE_N = 32   # Columns per block  
    BLOCK_SIZE_K = 32   # K dimension per block
    
    # Calculate grid size
    num_blocks_m = (n_rows + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M
    num_blocks_n = (n_cols + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N
    grid_size = num_blocks_m * num_blocks_n
    
    # Create output tensor
    output = torch.empty(n_rows, n_cols, dtype=torch.bfloat16, device=x.device)
    
    # Launch kernel
    linear_kernel[grid_size](
        x_ptr=x,
        weight_ptr=weight,
        output_ptr=output,
        n_cols=n_cols,
        n_rows=n_rows,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
    )
    
    return output

# Replacement function
def replacement_func():
    return optimized_linear