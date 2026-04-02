import torch
import triton
import triton.language as tl

def pattern(a, b):
    """
    Match matrix multiplication pattern from attention computation
    This matches torch.matmul(a, b) where a and b are attention tensors
    """
    result = torch.matmul(a, b)
    return result

def replacement_args(a, b):
    return (a, b)

@triton.jit
def simple_matmul_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    """
    Simple matrix multiplication kernel using Triton
    Optimized for the attention computation pattern
    """
    # Program ID allocation
    pid = tl.program_id(0)
    
    # Total programs calculation
    num_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_k = tl.cdiv(K, BLOCK_SIZE_K)
    total_programs = num_m * num_n * num_k
    
    # Check if this program should execute
    if pid >= total_programs:
        return
    
    # Calculate program coordinates
    pid_m = pid // (num_n * num_k)
    remainder = pid % (num_n * num_k)
    pid_n = remainder // num_k
    pid_k = remainder % num_k
    
    # Calculate block offsets
    m_offset = pid_m * BLOCK_SIZE_M
    n_offset = pid_n * BLOCK_SIZE_N
    k_offset = pid_k * BLOCK_SIZE_K
    
    # Create offsets within the block
    m_offsets = m_offset + tl.arange(0, BLOCK_SIZE_M)
    n_offsets = n_offset + tl.arange(0, BLOCK_SIZE_N)
    k_offsets = k_offset + tl.arange(0, BLOCK_SIZE_K)
    
    # Create masks
    m_mask = m_offsets < M
    n_mask = n_offsets < N
    k_mask = k_offsets < K
    
    # Initialize accumulator with zeros
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    # Main computation loop
    for k in range(k_offset, min(k_offset + K, k_offset + BLOCK_SIZE_K)):
        k_idx = k - k_offset
        
        # Load A block: [BLOCK_SIZE_M, BLOCK_SIZE_K]
        a_block = tl.load(
            a_ptr + m_offsets[:, None] * K + k,
            mask=m_mask[:, None] & (k_idx < BLOCK_SIZE_K),
            other=0.0
        )
        
        # Load B block: [BLOCK_SIZE_K, BLOCK_SIZE_N]  
        b_block = tl.load(
            b_ptr + k * N + n_offsets[None, :],
            mask=(k_idx < BLOCK_SIZE_K) & n_mask[None, :],
            other=0.0
        )
        
        # Matrix multiply and accumulate
        accumulator += tl.dot(a_block, b_block.to(tl.float32))
    
    # Store result
    tl.store(
        c_ptr + m_offsets[:, None] * N + n_offsets[None, :],
        accumulator.to(tl.float16),
        mask=m_mask[:, None] & n_mask[None, :]
    )

@torch.fx.wrap
def optimized_matmul(a, b):
    """
    Optimized matrix multiplication using Triton kernel
    Falls back to regular matmul for unsupported cases
    """
    # Get input shapes
    shape_a = a.shape
    shape_b = b.shape
    
    # Only optimize for 4D tensors (attention pattern)
    if len(shape_a) == 4 and len(shape_b) == 4:
        batch, heads, seq_len_a, dim_a = shape_a
        batch_b, heads_b, seq_len_b, dim_b = shape_b
        
        # Check compatibility
        if (batch == batch_b and heads == heads_b and 
            seq_len_a == seq_len_b and dim_a == seq_len_b):
            
            # Optimize each batch/head combination
            results = []
            
            for i in range(batch):
                for j in range(heads):
                    # Extract 2D matrices for this batch/head combination
                    a_2d = a[i, j]  # Shape: [seq_len, dim_a]
                    b_2d = b[i, j]  # Shape: [seq_len_b, dim_b]
                    
                    M, K = a_2d.shape
                    N = dim_b
                    
                    # Skip if dimensions are too small
                    if M <= 0 or N <= 0 or K <= 0:
                        result_2d = torch.matmul(a_2d, b_2d)
                    else:
                        # Allocate output
                        result_2d = torch.empty((M, N), dtype=a.dtype, device=a.device)
                        
                        # Choose block sizes based on matrix dimensions
                        BLOCK_SIZE_M = min(32, M)
                        BLOCK_SIZE_N = min(32, N)
                        BLOCK_SIZE_K = min(32, K)
                        
                        # Calculate grid size
                        num_m = (M + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M
                        num_n = (N + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N
                        num_k = (K + BLOCK_SIZE_K - 1) // BLOCK_SIZE_K
                        grid_size = num_m * num_n * num_k
                        
                        # Launch kernel only if we have a reasonable grid size
                        if grid_size > 0 and grid_size <= 1024:  # Limit to avoid too many blocks
                            simple_matmul_kernel[(grid_size,)](
                                a_ptr=a_2d,
                                b_ptr=b_2d,
                                c_ptr=result_2d,
                                M=M,
                                N=N,
                                K=K,
                                BLOCK_SIZE_M=BLOCK_SIZE_M,
                                BLOCK_SIZE_N=BLOCK_SIZE_N,
                                BLOCK_SIZE_K=BLOCK_SIZE_K,
                            )
                        else:
                            # Fall back to regular matmul for large/small matrices
                            result_2d = torch.matmul(a_2d, b_2d)
                    
                    results.append(result_2d)
            
            # Reshape back to original format
            return torch.stack(results).reshape(batch, heads, M, dim_b)
    
    # Fallback to regular matmul
    return torch.matmul(a, b)

def replacement_func():
    """Return optimized matmul function"""
    return optimized_matmul