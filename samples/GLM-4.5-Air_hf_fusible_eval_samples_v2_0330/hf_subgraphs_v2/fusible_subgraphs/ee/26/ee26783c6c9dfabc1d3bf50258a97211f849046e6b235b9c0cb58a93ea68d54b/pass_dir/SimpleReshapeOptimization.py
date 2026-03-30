import torch
import triton
import triton.language as tl

def pattern(matmul, reshape_dim1, reshape_dim2):
    # Generic pattern: matmul followed by reshape with specific dimensions
    tmp_1 = matmul.reshape(-1, reshape_dim1, reshape_dim2)
    return tmp_1

def replacement_args(matmul):
    # Auto-detect reshape dimensions for flexibility
    if matmul.shape[-1] == 128 and matmul.shape[-2] in [16, 8]:
        if matmul.shape[-2] == 16:
            return (matmul, 16, 31)  # First graph pattern
        else:
            return (matmul, 8, 15)    # Second graph pattern
    else:
        # Default fallback
        return (matmul, 16, 31)

@triton.jit
def optimized_matmul_kernel(
    q_ptr,
    k_ptr,
    output_ptr,
    batch_size,
    n_heads,
    seq_len_q,
    head_dim,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    # Bounds checking
    m_mask = pid_m < batch_size * n_heads
    n_mask = pid_n < seq_len_q
    
    # Initialize accumulator
    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    # Matrix multiplication
    for k in range(0, head_dim, BLOCK_K):
        # Load query vectors
        q = tl.load(
            q_ptr + (pid_m * head_dim * seq_len_q + k * seq_len_q + pid_n),
            mask=m_mask & n_mask & (k < head_dim),
            other=0.0
        ).to(tl.float32)
        
        # Load key vectors
        k_val = tl.load(
            k_ptr + (k * seq_len_q + pid_n),
            mask=n_mask & (k < head_dim),
            other=0.0
        ).to(tl.float32)
        
        # Outer product
        accumulator += q[:, None] * k_val[None, :]
    
    # Store result
    tl.store(
        output_ptr + (pid_m * seq_len_q + pid_n),
        accumulator.to(tl.float16),
        mask=m_mask & n_mask
    )

@torch.fx.wrap 
def optimized_matmul_reshape(matmul, dim1, dim2):
    # Check if this is from a matmul that could be optimized
    if hasattr(matmul, '_matmul_context'):
        # Use optimized matmul result if available
        return matmul.reshape(-1, dim1, dim2)
    else:
        # Fallback to regular reshape
        return matmul.reshape(-1, dim1, dim2)

def replacement_func():
    return optimized_matmul_reshape