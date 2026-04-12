import torch
import triton
import triton.language as tl

# Pattern matching function
def pattern(in_0, in_1):
    """
    Match: scalar_multiply -> softmax -> matmul -> permute(0,2,1)
    This is an attention-like computation pattern that can be fused efficiently.
    """
    tmp_0 = 0.0625 * in_0
    tmp_1 = torch.nn.functional.softmax(tmp_0, dim=-1)
    matmul = torch.matmul(tmp_1, in_1)
    tmp_3 = matmul.permute(0, 2, 1)
    return tmp_3

# Argument extraction function  
def replacement_args(in_0, in_1):
    return (in_0, in_1)

# Triton kernel for fused attention computation
@triton.jit
def fused_attention_kernel(
    in_0_ptr, 
    in_1_ptr, 
    out_ptr,
    batch_size,
    seq_len_k,
    seq_len_q,
    value_dim,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr, 
    BLOCK_SIZE_K: tl.constexpr,
):
    """
    Fused kernel for: scale * softmax(in_0) @ in_1, then transpose(0,2,1)
    
    Input shapes:
    - in_0: [batch_size, seq_len_k, seq_len_q]
    - in_1: [batch_size, seq_len_q, value_dim] 
    Output:
    - out: [batch_size, value_dim, seq_len_k]
    """
    pid = tl.program_id(0)
    
    # Split work across output dimensions (B, V, K) -> we split on V and K
    grid_value = (value_dim + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N
    grid_key = (seq_len_k + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M
    pid_value = pid % grid_value
    pid_key = pid // grid_value
    pid_value_local = pid_value * BLOCK_SIZE_N
    pid_key_local = pid_key * BLOCK_SIZE_M
    
    batch_idx = tl.program_id(1)
    
    # Compute base pointers
    in_0_base = in_0_ptr + batch_idx * seq_len_k * seq_len_q
    in_1_base = in_1_ptr + batch_idx * seq_len_q * value_dim
    out_base = out_ptr + batch_idx * value_dim * seq_len_k
    
    # Initialize output accumulator for this output position
    output_val = tl.zeros([BLOCK_SIZE_M], dtype=tl.float32)
    
    # Process query dimension (seq_len_q) in chunks for matmul
    for k in range(0, seq_len_q, BLOCK_SIZE_K):
        k_offset = k + tl.arange(0, BLOCK_SIZE_K)
        k_mask = k_offset < seq_len_q
        
        # Load current key segment from in_0 (K dimension)
        k_ptrs = in_0_base + pid_key_local * seq_len_q + k_offset
        k_vals = tl.load(k_ptrs, mask=k_mask, other=0.0)
        
        # Load current value segment from in_1  
        v_ptrs = in_1_base + k_offset * value_dim + pid_value_local
        v_vals = tl.load(v_ptrs, mask=k_mask, other=0.0)
        
        # Compute attention scores with scaling: k @ v_weights * 0.0625
        # Then apply softmax
        scores = k_vals * 0.0625
        max_score = tl.maximum(tl.max(scores), 0.0)  # Simplified max for stability
        exp_scores = tl.exp(scores - max_score)
        softmax_weights = exp_scores / (tl.sum(exp_scores) + 1e-8)
        
        # Weighted sum: softmax_weights @ v_vals
        weighted_v = tl.sum(softmax_weights * v_vals, axis=0)
        
        # Accumulate to output
        output_val += weighted_v
    
    # Store final result
    out_ptr_local = out_base + pid_value_local * seq_len_k + pid_key_local
    tl.store(out_ptr_local, output_val, mask=tl.arange(0, BLOCK_SIZE_M) < seq_len_k)

# Kernel wrapper
@torch.fx.wrap
def fused_attention_triton(in_0, in_1):
    # Get tensor dimensions
    batch_size, seq_len_k, seq_len_q = in_0.shape
    _, _, value_dim = in_1.shape
    
    # Output should be [batch_size, value_dim, seq_len_k]
    out = torch.empty((batch_size, value_dim, seq_len_k), dtype=in_0.dtype, device=in_0.device)
    
    # Choose block sizes based on tensor dimensions for good occupancy
    BLOCK_SIZE_M = 256  # Process seq_len_k in chunks of this size
    BLOCK_SIZE_N = 64   # Process value_dim in chunks of this size  
    BLOCK_SIZE_K = 32   # Process seq_len_q in chunks of this size
    
    # Calculate grid size for output dimensions (value_dim and seq_len_k)
    grid_size_value = (value_dim + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N
    grid_size_key = (seq_len_k + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M
    grid_size = grid_size_value * grid_size_key
    
    # Launch kernel with 2D grid: [grid_size, batch_size]
    fused_attention_kernel[(grid_size, batch_size)](
        in_0,
        in_1, 
        out,
        batch_size,
        seq_len_k,
        seq_len_q,
        value_dim,
        BLOCK_SIZE_M,
        BLOCK_SIZE_N,
        BLOCK_SIZE_K,
    )
    
    return out

# Replacement function
def replacement_func():
    return fused_attention_triton