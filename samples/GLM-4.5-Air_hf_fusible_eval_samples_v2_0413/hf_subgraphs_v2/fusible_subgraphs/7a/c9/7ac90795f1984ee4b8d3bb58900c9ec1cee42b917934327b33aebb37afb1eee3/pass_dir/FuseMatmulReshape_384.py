import torch
import triton
import triton.language as tl

def pattern(in_1, in_0):
    matmul = torch.matmul(in_1, in_0)
    tmp_1 = torch.reshape(matmul, [-1, 384])
    return tmp_1

def replacement_args(in_1, in_0):
    return (in_1, in_0, 384)

@triton.jit
def matmul_kernel_384(
    x_ptr,
    y_ptr,
    output_ptr,
    batch_size,
    m, n, k,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    pid = tl.program_id(0)
    
    # Calculate grid dimensions
    grid_m = (m + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M
    grid_n = (n + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N
    grid_b = batch_size
    
    batch_idx = pid % grid_b
    pid_mn = pid // grid_b
    pid_m = pid_mn % grid_m
    pid_n = pid_mn // grid_m
    
    # Skip out-of-bound programs
    if (pid_m * BLOCK_SIZE_M >= m or pid_n * BLOCK_SIZE_N >= n or 
        batch_idx >= batch_size):
        return
    
    # Compute matrix addresses for current batch
    x_offset = batch_idx * (n * k) + pid_m * BLOCK_SIZE_M
    y_offset = batch_idx * (m * k) + pid_n * BLOCK_SIZE_N
    z_offset = batch_idx * (m * n) + pid_m * BLOCK_SIZE_M + pid_n * BLOCK_SIZE_N * m
    
    a_ptr = x_ptr + x_offset
    b_ptr = y_ptr + y_offset
    c_ptr = output_ptr + z_offset
    
    # Initialize accumulator
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    # Matrix multiplication loop
    for k_off in range(0, k, BLOCK_SIZE_K):
        # Load matrix A and B tiles
        mask = k_off + tl.arange(0, BLOCK_SIZE_K) < k
        a = tl.load(a_ptr + k_off + tl.arange(0, BLOCK_SIZE_K), mask=mask, other=0.0)
        b = tl.load(b_ptr + k_off + tl.arange(0, BLOCK_SIZE_K)[:, None] * m, 
                   mask=mask[:, None], other=0.0)
        
        # Accumulate results
        acc += tl.dot(a, b, allow_tf32=False)
    
    # Store results with proper masking
    mask = (tl.arange(0, BLOCK_SIZE_M) < m - pid_m * BLOCK_SIZE_M) & \
           (tl.arange(0, BLOCK_SIZE_N)[:, None] < n - pid_n * BLOCK_SIZE_N)
    tl.store(c_ptr + tl.arange(0, BLOCK_SIZE_M)[:, None] * m, 
             acc.to(tl.float16), mask=mask)

@torch.fx.wrap
def triton_matmul_reshape_384(x, y, target_dim):
    # Get input dimensions
    x_shape = x.shape
    y_shape = y.shape
    
    if len(x_shape) == 3 and len(y_shape) == 3:
        batch_size, m, k = x_shape
        _, _, n = y_shape
        
        # Compute output shape for reshape(-1, 384)
        total_elements = batch_size * m * n
        output_rows = total_elements // 384
        
        # Create output tensor
        output = torch.empty(output_rows, 384, dtype=x.dtype, device=x.device)
        
        # Configure block sizes optimized for larger matrices
        BLOCK_SIZE_M = 64  # Larger blocks for bigger matrices
        BLOCK_SIZE_N = 64
        BLOCK_SIZE_K = 32
        
        # Calculate grid size
        total_programs = batch_size * ((m + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M) * \
                        ((n + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N)
        
        # Launch kernel
        matmul_kernel_384[total_programs](
            x, y, output,
            batch_size, m, n, k,
            BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K
        )
        
        return output
    else:
        # Fallback for non-batched cases
        matmul_result = torch.matmul(x, y)
        return torch.reshape(matmul_result, [-1, target_dim])

def replacement_func():
    return triton_matmul_reshape_384