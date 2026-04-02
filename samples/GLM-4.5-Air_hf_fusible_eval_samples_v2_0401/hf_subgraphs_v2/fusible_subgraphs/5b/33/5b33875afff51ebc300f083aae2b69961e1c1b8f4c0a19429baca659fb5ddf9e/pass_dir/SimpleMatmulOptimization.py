import torch
import triton
import triton.language as tl

def pattern(a, b):
    """
    Simple matrix multiplication pattern
    """
    return torch.matmul(a, b)

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
    BLOCK_SIZE: tl.constexpr,
):
    """
    Simple matrix multiplication kernel
    Optimized for attention computation patterns
    """
    pid_m = tl.program_id(0)
    m_offsets = pid_m * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    m_mask = m_offsets < M
    
    # Initialize accumulator
    accumulator = tl.zeros((BLOCK_SIZE, N), dtype=tl.float32)
    
    for k in range(0, K):
        a = tl.load(a_ptr + m_offsets[:, None] * K + k, mask=m_mask[:, None], other=0.0)
        b = tl.load(b_ptr + k * N + tl.arange(0, N)[None, :], mask=(k < K), other=0.0)
        accumulator += a * b.to(tl.float32)
    
    tl.store(c_ptr + m_offsets[:, None] * N + tl.arange(0, N)[None, :],
             accumulator.to(tl.float16), mask=m_mask[:, None])

@torch.fx.wrap
def optimized_matmul(a, b):
    """
    Optimized matrix multiplication using Triton
    Falls back to regular matmul for unsupported cases
    """
    # Try to use optimized version for 4D tensors (attention pattern)
    if len(a.shape) == 4 and len(b.shape) == 4:
        batch, heads, seq_len, dim_a = a.shape
        batch_b, heads_b, seq_len_b, dim_b = b.shape
        
        if batch == batch_b and heads == heads_b and seq_len == seq_len_b and dim_a == seq_len_b:
            # Reshape to 2D for each batch/head combination
            a_2d = a.reshape(batch * heads, seq_len, dim_a)
            b_2d = b.reshape(batch * heads, seq_len_b, dim_b)
            
            # Use simple matmul for each 2D slice
            results = []
            for i in range(batch * heads):
                result = torch.zeros(seq_len, dim_b, dtype=a.dtype, device=a.device)
                M, K = seq_len, dim_a
                N = dim_b
                
                if M > 0 and N > 0 and K > 0:
                    BLOCK_SIZE = min(64, M)
                    grid = (M + BLOCK_SIZE - 1) // BLOCK_SIZE
                    
                    simple_matmul_kernel[grid](
                        a_ptr=a_2d[i],
                        b_ptr=b_2d[i], 
                        c_ptr=result,
                        M=M,
                        N=N,
                        K=K,
                        BLOCK_SIZE=BLOCK_SIZE,
                    )
                else:
                    result = torch.matmul(a_2d[i], b_2d[i])
                
                results.append(result)
            
            return torch.stack(results).reshape(batch, heads, seq_len, dim_b)
    
    # Fallback to regular matmul
    return torch.matmul(a, b)

def replacement_func():
    """Return optimized matmul function"""
    return optimized_matmul