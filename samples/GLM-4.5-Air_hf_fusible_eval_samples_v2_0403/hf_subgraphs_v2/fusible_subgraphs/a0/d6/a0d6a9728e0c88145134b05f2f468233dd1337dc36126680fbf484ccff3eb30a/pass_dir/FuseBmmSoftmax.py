import torch
import triton
import triton.language as tl

# Pattern matching for BMM -> Softmax fusion
def pattern(a, b):
    # First BMM: Query @ Key.transpose
    bmm_result = torch.bmm(a, b)
    # Softmax activation
    softmax_result = torch.nn.functional.softmax(bmm_result, dim=-1)
    return bmm_result, softmax_result

# Arguments needed for the replacement kernel
def replacement_args(a, b):
    return (a, b)

# Optimized fused BMM-Softmax kernel
@triton.jit
def fused_bmm_softmax_kernel(
    query_ptr, key_ptr, out_ptr,
    batch_size, seq_len_q, seq_len_k, dim_k,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr
):
    # Compute program IDs
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    # Range of elements to reduce
    k_offset = tl.program_id(2) * BLOCK_SIZE_K
    
    # Compute offsets
    m_offset = pid_m * BLOCK_SIZE_M
    n_offset = pid_n * BLOCK_SIZE_N
    
    # Initialize accumulator for BMM
    accumulator = 0.0
    
    # Loop over k dimension for matrix multiplication
    for k in range(0, seq_len_k, BLOCK_SIZE_K):
        # Load query slice
        query_offset = m_offset * (seq_len_k * dim_k) + k * dim_k
        query_ptrs = query_offset + tl.arange(0, dim_k)
        query = tl.load(query_ptrs, mask=query_offset + tl.arange(0, dim_k) < batch_size * seq_len_q * dim_k, other=0.0).to(tl.float32)
        
        # Load key slice
        key_offset = (k + k_offset) * seq_len_k + n_offset
        key_ptrs = key_offset + tl.arange(0, dim_k)
        key = tl.load(key_ptrs, mask=key_offset + tl.arange(0, dim_k) < batch_size * seq_len_k * dim_k, other=0.0).to(tl.float32)
        
        # Matrix multiply and accumulate
        accumulator += tl.sum(query * key)
    
    # Apply softmax to the computed BMM result
    max_val = tl.maximum(accumulator, -10000.0)  # For numerical stability
    softmax_exp = tl.exp(accumulator - max_val)
    softmax_result = softmax_exp / (softmax_exp + 1e-6)  # Add epsilon for stability
    
    # Store result
    out_offset = m_offset * seq_len_k + n_offset
    out_ptrs = out_offset + tl.arange(0, 1)  # seq_len_k dimension
    tl.store(out_ptrs, softmax_result, mask=out_offset + tl.arange(0, 1) < batch_size * seq_len_q * seq_len_k)

# Kernel wrapper with proper dimension handling
@torch.fx.wrap
def fused_bmm_softmax(query, key):
    # Get tensor shapes
    batch_size, seq_len_q, dim_q = query.shape
    batch_size_k, seq_len_k, dim_k = key.shape
    
    # Verify dimensions
    assert dim_q == dim_k, "Query and key dimension mismatch"
    assert batch_size == batch_size_k, "Batch size mismatch"
    
    # Output shape is [batch_size, seq_len_q, seq_len_k]
    output_shape = (batch_size, seq_len_q, seq_len_k)
    output = torch.empty(output_shape, dtype=query.dtype, device=query.device)
    
    # Triton kernel launch configuration
    # For typical attention patterns, optimize for small seq_len with larger dim
    BLOCK_SIZE_M = 1
    BLOCK_SIZE_N = 1
    BLOCK_SIZE_K = 64
    
    # Grid calculation
    num_blocks_m = (seq_len_q + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M
    num_blocks_n = (seq_len_k + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N
    num_blocks_k = (dim_k + BLOCK_SIZE_K - 1) // BLOCK_SIZE_K
    
    # Launch kernel
    fused_bmm_softmax_kernel[(num_blocks_m, num_blocks_n, num_blocks_k)](
        query=query,
        key=key,
        out=output,
        batch_size=batch_size,
        seq_len_q=seq_len_q,
        seq_len_k=seq_len_k,
        dim_k=dim_k,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K
    )
    
    return output

# Replacement function (returns function reference, not a call)
def replacement_func():
    return fused_bmm_softmax