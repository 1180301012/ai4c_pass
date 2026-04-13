import torch
import triton
import triton.language as tl

def pattern(in_1, in_0):
    matmul = torch.matmul(in_1, in_0)
    tmp_1 = torch.reshape(matmul, [-1, 128])
    return tmp_1

def replacement_args(in_1, in_0):
    return (in_1, in_0, 128)

@triton.jit
def matmul_kernel_optimized(
    x_ptr,
    y_ptr,
    output_ptr,
    batch_size,
    m, n, k,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    # Calculate program indices
    programs_per_m = (n + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N
    pid = tl.program_id(0)
    pid_m = pid % programs_per_m
    pid_n = pid // programs_per_m
    
    if pid_m * BLOCK_SIZE_M >= m or pid_n * BLOCK_SIZE_N >= n:
        return
    
    # Process across batch dimension in groups
    batch_groups = min(4, batch_size)  # Process up to 4 batches per kernel for better utilization
    remaining_batches = batch_size
    
    for batch_start in range(0, batch_size, batch_groups):
        batch_end = min(batch_start + batch_groups, batch_size)
        current_batch_size = batch_end - batch_start
        
        # Matrix addresses for this batch group
        x_offset = batch_start * (n * k)
        y_offset = batch_start * (k * m)
        
        # Process each batch in the group
        for batch_idx in range(current_batch_size):
            # Compute pointers for this batch
            a_ptr = x_ptr + x_offset + batch_idx * (n * k) + pid_n * BLOCK_SIZE_N * k
            b_ptr = y_ptr + y_offset + batch_idx * (k * m) + pid_m * BLOCK_SIZE_M
            
            c_ptr = output_ptr + (pid_m * BLOCK_SIZE_M + pid_n * BLOCK_SIZE_N * m + 
                                batch_idx * (m * n // 128 * 128))
            
            acc = tl.zeros((BLOCK_SIZE_N, BLOCK_SIZE_M), dtype=tl.float32)
            
            # Main computation loop
            for k_off in range(0, k, BLOCK_SIZE_K):
                a = tl.load(
                    a_ptr + k_off + tl.arange(0, BLOCK_SIZE_K),
                    mask=k_off + tl.arange(0, BLOCK_SIZE_K) < k,
                    other=0.0
                )
                b = tl.load(
                    b_ptr + (k_off * n + tl.arange(0, BLOCK_SIZE_K)[:, None]),
                    mask=(k_off + tl.arange(0, BLOCK_SIZE_K)[:, None]) < k,
                    other=0.0
                )
                acc += tl.dot(a, b)
            
            # Store result
            mask = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) < n and \
                   (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) < m
            tl.store(
                c_ptr + (tl.arange(0, BLOCK_SIZE_N)[:, None] * m + tl.arange(0, BLOCK_SIZE_M)),
                acc.to(tl.float16),
                mask=mask
            )

@torch.fx.wrap
def triton_matmul_reshape_128(x, y, target_dim):
    # Get input shapes
    x_shape = x.shape
    y_shape = y.shape
    
    if len(x_shape) == 3 and len(y_shape) == 3:
        batch_size, m, k = x_shape
        _, _, n = y_shape
        
        # Calculate output shape for reshape(-1, 128)
        total_elements = batch_size * m * n
        output_rows = total_elements // 128
        
        # Create output tensor
        output = torch.empty(output_rows, 128, dtype=x.dtype, device=x.device)
        
        # Optimize block sizes for better performance with target_dim=128
        BLOCK_SIZE_M = 32
        BLOCK_SIZE_N = 32
        BLOCK_SIZE_K = 32
        
        # Calculate grid size with batch processing
        grid_size = (m + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M * \
                   (n + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N
        
        # Launch kernel
        matmul_kernel_optimized[grid_size * batch_size](
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
    return triton_matmul_reshape_128