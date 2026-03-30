import torch
import triton
import triton.language as tl

def pattern(in_1, in_3):
    # Initial query-key matmul that starts the attention mechanism
    matmul = in_1 @ in_3
    tmp_1 = matmul.reshape(-1, 16, 31)  # This reshape is immediately used in bias computation
    return tmp_1

def replacement_args(in_1, in_3):
    return (in_1, in_3)

@triton.jit
def optimized_matmul_kernel(
    q_ptr,
    k_ptr,
    output_ptr,
    batch_size,
    n_heads,
    head_dim,
    seq_len_k,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    # Program IDs for matrix multiplication
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    # Get global IDs
    batch_id = pid_m // (n_heads * seq_len_k)
    head_id = (pid_m % (n_heads * seq_len_k)) // seq_len_k
    seq_id = pid_m % seq_len_k
    
    # Bounds checking
    m_mask = pid_m < batch_size * n_heads * seq_len_k
    n_mask = pid_n < 16  # Target reshape dimension 1
    k_mask = pid_n < head_dim
    
    # Initialize accumulator
    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    # Block-level matrix multiplication with tiling
    for k in range(0, head_dim, BLOCK_K):
        # Load query vectors (batch, head, seq_q, head_dim)
        q_offset = batch_id * n_heads * head_dim * seq_len_k + head_id * head_dim * seq_len_k + seq_id * head_dim + k
        q = tl.load(
            q_ptr + q_offset,
            mask=(k < head_dim) & (seq_id < seq_len_k),
            other=0.0
        ).to(tl.float32)
        
        # Load key vectors (head_dim, seq_len_k)
        # We need to load multiple key vectors for the blocked k dimension
        for kk in range(BLOCK_K):
            k_offset = (k + kk) * seq_len_k + pid_n
            k_val = tl.load(
                k_ptr + k_offset,
                mask=(k + kk < head_dim) & (pid_n < seq_len_k),
                other=0.0
            ).to(tl.float32)
            
            # Outer product and accumulate
            accumulator += q[:, None] * k_val[None, :]
    
    # Store result directly in the reshaped format [batch*n_heads, 16, 31]
    output_batch_offset = batch_id * n_heads * 16 * 31
    output_head_offset = head_id * 16 * 31
    output_offset = output_batch_offset + output_head_offset + pid_n * 31 + tl.arange(0, BLOCK_M)
    
    tl.store(
        output_ptr + output_offset,
        accumulator.to(tl.float16),
        mask=(pid_m < batch_size * n_heads * seq_len_k) & (tl.arange(0, BLOCK_M) < 31)
    )

@torch.fx.wrap
def optimized_qk_matmul(in_1, in_3):
    # Input shapes: in_1 [4, 16, 16, 128], in_3 [128, 31]
    batch_size = in_1.shape[0]
    n_heads = in_1.shape[1]
    head_dim = in_1.shape[3]
    seq_len_k = in_3.shape[1]  # This should be 31 based on the reshape
    
    # Output shape should be [4*16*16, 16, 31] = [1024, 16, 31]
    output_shape = [batch_size * n_heads * in_1.shape[2], 16, 31]
    
    out = torch.zeros(output_shape, dtype=torch.float16, device=in_1.device)
    
    # Kernel configuration - optimized for transformer sizes
    BLOCK_M = 32
    BLOCK_N = 16
    BLOCK_K = 32
    
    # Calculate grid dimensions
    grid_m = batch_size * n_heads * in_1.shape[2] // BLOCK_M
    grid_n = 16 // BLOCK_N
    
    optimized_matmul_kernel[(grid_m, grid_n)](
        q_ptr=in_1,
        k_ptr=in_3,
        output_ptr=out,
        batch_size=batch_size,
        n_heads=n_heads,
        head_dim=head_dim,
        seq_len_k=seq_len_k,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
    )
    
    return out

def replacement_func():
    return optimized_qk_matmul