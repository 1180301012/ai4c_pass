import torch
import triton
import triton.language as tl

def pattern(in_0, in_1):
    tmp_0 = 0.0625 * in_0
    tmp_1 = torch.nn.functional.softmax(tmp_0, dim=-1)
    matmul = torch.matmul(tmp_1, in_1)
    tmp_3 = matmul.permute(0, 2, 1)
    return tmp_3

def replacement_args(in_0, in_1):
    return (in_0, in_1)

@triton.jit
def end_to_end_fusion_kernel(
    a_ptr,
    b_ptr,
    out_ptr,
    batch_size,
    seq_len,
    num_heads,
    hidden_dim,
    scale_factor: tl.constexpr,
    BLOCK_SOFTMAX: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    # Grid setup
    batch_idx = tl.program_id(0)
    head_idx = tl.program_id(1)
    n_idx = tl.program_id(2)
    m_idx = tl.program_id(3)
    
    # Process softmax on a's sequence
    a_start = a_ptr + (batch_idx * seq_len * num_heads + head_idx * seq_len)
    
    # Load softmax input
    a_seq = tl.load(a_start + tl.arange(0, seq_len), mask=tl.arange(0, seq_len) < seq_len, other=float('-inf'))
    a_scaled = scale_factor * a_seq
    max_val = tl.max(a_scaled)
    exp_vals = tl.exp(a_scaled - max_val)
    sum_exp = tl.sum(exp_vals)
    softmax_vals = exp_vals / sum_exp
    
    # Process matmul with output permutation
    batch_offset = batch_idx * hidden_dim * seq_len
    a_mat_start = batch_offset + head_idx * seq_len * hidden_dim + n_idx * BLOCK_K
    b_mat_start = batch_offset + tl.arange(0, BLOCK_K)[:, None] * hidden_dim + (m_idx * BLOCK_N + tl.arange(0, BLOCK_N))[None, :]
    out_start = batch_offset + (m_idx * BLOCK_N + tl.arange(0, BLOCK_N))[:, None] * seq_len + (head_idx * seq_len + tl.arange(0, seq_len))[None, :]
    
    # Initialize accumulator
    accumulator = tl.zeros((BLOCK_N, seq_len), dtype=tl.float32)
    
    # Loop over k dimension (num_heads dimension)
    for k_idx in range(0, seq_len, BLOCK_K):
        k_mask = k_idx < seq_len
        
        # Load a values (softmax output)
        a_offset = a_mat_start + (tl.arange(0, BLOCK_K) * seq_len)
        a_vals = tl.load(a_ptr + a_offset, mask=tl.arange(0, BLOCK_K) < (seq_len - k_idx), other=0.0)
        
        # Load b values
        b_offset = b_mat_start + (k_idx + tl.arange(0, BLOCK_K))[:, None] * hidden_dim
        b_mask = (k_idx + tl.arange(0, BLOCK_K))[:, None] < num_heads
        b_vals = tl.load(b_ptr + b_offset, mask=b_mask, other=0.0)
        
        # Matrix multiplication with broadcast
        for i in range(BLOCK_N):
            accumulator[i] += a_vals * b_vals[:, i]
    
    # Store result with proper masking
    out_mask = (tl.arange(0, BLOCK_N)[:, None] < (hidden_dim - m_idx * BLOCK_N)) & (tl.arange(0, seq_len)[None, :] < seq_len)
    tl.store(out_ptr + out_start, accumulator.T, mask=out_mask)

@torch.fx.wrap
def end_to_end_fusion(a, b):
    batch_size, seq_len, num_heads = a.shape
    _, hidden_dim_in, hidden_dim_out = b.shape
    
    # Scale factor from original computation
    scale_factor = 0.0625
    
    # Output is [batch_size, hidden_dim_out, seq_len]
    output = torch.empty((batch_size, hidden_dim_out, seq_len), dtype=a.dtype, device=a.device)
    
    # Block sizes
    BLOCK_SOFTMAX = 32
    BLOCK_M = 128
    BLOCK_N = 64
    BLOCK_K = 32
    
    # Calculate grid dimensions
    grid_m = (hidden_dim_out + BLOCK_M - 1) // BLOCK_M
    grid_n = (hidden_dim_in + BLOCK_K - 1) // BLOCK_K
    
    # Launch kernel
    end_to_end_fusion_kernel[(batch_size, num_heads, grid_n, grid_m)](
        a_ptr=a,
        b_ptr=b,
        out_ptr=output,
        batch_size=batch_size,
        seq_len=seq_len,
        num_heads=num_heads,
        hidden_dim=hidden_dim_out,
        scale_factor=scale_factor,
        BLOCK_SOFTMAX=BLOCK_SOFTMAX,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
    )
    
    return output

def replacement_func():
    return end_to_end_fusion