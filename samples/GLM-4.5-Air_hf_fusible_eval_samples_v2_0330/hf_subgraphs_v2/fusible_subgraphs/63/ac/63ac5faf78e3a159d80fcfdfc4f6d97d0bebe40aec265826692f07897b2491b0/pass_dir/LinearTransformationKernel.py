import torch
import triton
import triton.language as tl

# Pattern matching for the linear transformation operation
def linear_pattern(weight, input):
    """
    Match the linear transformation pattern: output = input @ weight.T
    This corresponds to torch.nn.functional.linear(input, weight, None)
    """
    output = input @ weight.T
    return output

# Extract arguments for the replacement kernel
def replacement_args(weight, input):
    return (weight, input)

# Triton kernel for optimized matrix multiplication (linear transformation)
@triton.jit
def linear_kernel(
    input_ptr,
    weight_ptr,
    output_ptr,
    M,
    N,
    K,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    """
    Optimized matrix multiplication kernel: C = A @ B.T
    Where A is input (M x K), B is weight (N x K), C is output (M x N)
    """
    # Program identifiers for 2D grid
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)
    
    # Compute ranges for this program
    m_start = pid_m * BLOCK_SIZE_M
    n_start = pid_n * BLOCK_SIZE_N
    
    # Create offsets within the block
    offs_m = m_start + tl.arange(0, BLOCK_SIZE_M)
    offs_n = n_start + tl.arange(0, BLOCK_SIZE_N)
    
    # Initialize accumulator
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    # Loop over K dimension
    for k in range(0, K, BLOCK_SIZE_K):
        # Compute current K block bounds
        k_start = k
        k_end = tl.minimum(k + BLOCK_SIZE_K, K)
        
        # Create 2D offsets within the block
        offs_k = tl.arange(k_start, k_end)
        
        # Load input block (A): shape (BLOCK_SIZE_M, BLOCK_SIZE_K)
        input_ptrs = input_ptr + (offs_m[:, None] * K + offs_k[None, :])
        input_block = tl.load(input_ptrs, mask=(offs_m[:, None] < M)[:, None] & (offs_k[None, :] < K), other=0.0)
        
        # Load weight block (B): shape (BLOCK_SIZE_N, BLOCK_SIZE_K)
        weight_ptrs = weight_ptr + (offs_n[:, None] * K + offs_k[None, :])
        weight_block = tl.load(weight_ptrs, mask=(offs_n[:, None] < N)[:, None] & (offs_k[None, :] < K), other=0.0)
        
        # Matrix multiplication and accumulation
        accumulator += tl.dot(input_block, weight_block, trans_b=True)
    
    # Store output block
    output_ptrs = output_ptr + (offs_m[:, None] * N + offs_n[None, :])
    output_mask = (offs_m[:, None] < M)[:, None] & (offs_n[None, :] < N)
    tl.store(output_ptrs, accumulator, mask=output_mask)

@torch.fx.wrap
def optimized_linear_transform(input, weight):
    """
    Optimized version of torch.nn.functional.linear(input, weight, None)
    """
    # Get tensor dimensions
    M = input.shape[0]  # Batch size or input features
    K = input.shape[1]  # Input dimension
    N = weight.shape[0]  # Output dimension
    
    # Create output tensor
    output = torch.empty((M, N), dtype=input.dtype, device=input.device)
    
    # Optimized block sizes for different precision levels
    if input.dtype == torch.float16 or input.dtype == torch.bfloat16:
        BLOCK_SIZE_M = 64
        BLOCK_SIZE_N = 64
        BLOCK_SIZE_K = 32
    else:  # float32
        BLOCK_SIZE_M = 32
        BLOCK_SIZE_N = 32
        BLOCK_SIZE_K = 16
    
    # Calculate grid dimensions
    num_blocks_m = (M + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M
    num_blocks_n = (N + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N
    
    # Launch Triton kernel
    linear_kernel[(num_blocks_m, num_blocks_n)](
        input_ptr=input,
        weight_ptr=weight,
        output_ptr=output,
        M=M,
        N=N,
        K=K,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
    )
    
    return output

# Replacement function - must return a callable function reference
def replacement_func():
    return optimized_linear_transform