import torch
import triton
import triton.language as tl

# Pattern matching for the entire attention computation with bias
def pattern(q, k, v, bias, residual):
    # Original computation exactly as in model:
    matmul = q @ k
    tmp_1 = matmul.reshape(-1, 16, 31)
    tmp_2 = torch.nn.functional.pad(tmp_1, [0, 1], 'constant', None)
    tmp_3 = tmp_2.flatten(1)
    tmp_4 = torch.nn.functional.pad(tmp_3, [0, 15], 'constant', None)
    tmp_5 = tmp_4.reshape(-1, 17, 31)
    tmp_6 = tmp_5[(slice(None, None, None), slice(None, 16, None), slice(15, None, None))]
    tmp_7 = tmp_6.reshape(4, 16, 1, 16, 16)
    tmp_8 = tmp_7.expand(-1, -1, 16, -1, -1)
    tmp_9 = tmp_8.permute((0, 3, 1, 4, 2))
    attn_scores = tmp_9 + bias
    tmp_11 = attn_scores.reshape(4, 256, 256)
    tmp_12 = residual + tmp_11
    attn_output = tmp_12.softmax(dim = -1)
    matmul_1 = attn_output @ v
    output = matmul_1.transpose(-1, -2)
    
    # Return all observables for pattern matching
    return output


def replacement_args(q, k, v, bias, residual):
    return (q, k, v, bias, residual)


@triton.jit
def fused_attention_kernel(
    q_ptr, k_ptr, v_ptr, bias_ptr, residual_ptr,
    output_ptr,
    batch_size, seq_len, heads, head_dim, 
    seq_len2, heads2,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr
):
    # Compute block offsets
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    # Initialize pointers
    q_block_start = pid_m * BLOCK_SIZE_M * seq_len * head_dim
    k_block_start = pid_n * BLOCK_SIZE_N * seq_len2 * head_dim
    
    # Load Q and K blocks
    q_offsets = q_block_start + tl.arange(0, BLOCK_SIZE_M * seq_len * head_dim)
    k_offsets = k_block_start + tl.arange(0, BLOCK_SIZE_N * seq_len2 * head_dim)
    
    # Load bias for attention pattern
    bias_m = pid_m
    bias_n = pid_n
    bias_offset = (bias_m * seq_len + bias_n * seq_len2) * seq_len2
    bias_offsets = bias_offset + tl.arange(0, seq_len2)
    
    # Perform matmul with bias and attention pattern
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float16)
    
    for k in range(0, head_dim, BLOCK_SIZE_K):
        # Load Q and K blocks
        q_val = tl.load(q_ptr + q_offsets + k * BLOCK_SIZE_M * seq_len, mask=(k + tl.arange(0, BLOCK_SIZE_K)) < head_dim, other=0.0)
        k_val = tl.load(k_ptr + k_offsets + k * BLOCK_SIZE_N * seq_len2, mask=(k + tl.arange(0, BLOCK_SIZE_K)) < head_dim, other=0.0)
        
        # Compute GEMM
        q_reshaped = q_val.reshape(BLOCK_SIZE_M, seq_len, BLOCK_SIZE_K)
        k_reshaped = k_val.reshape(BLOCK_SIZE_N, seq_len2, BLOCK_SIZE_K)
        
        # Compute attention scores pattern
        for i in range(BLOCK_SIZE_M):
            for j in range(BLOCK_SIZE_N):
                # Simplified attention pattern implementation
                acc[i, j] += tl.sum(q_reshaped[i, :, :] * k_reshaped[j, :, :], axis=1)
    
    # Add relative position bias
    if bias_ptr is not None:
        bias_val = tl.load(bias_ptr + bias_offsets)
        # Apply bias pattern (simplified for demonstration)
        acc = acc + bias_val.reshape(1, BLOCK_SIZE_N)
    
    # Add residual and apply softmax
    if residual_ptr is not None:
        residual_val = tl.load(residual_ptr + pid_m * seq_len + pid_n, mask=(pid_m * seq_len + pid_n) < (batch_size * seq_len))
        acc = acc + residual_val
        
    # Apply softmax
    max_val = tl.max(acc, axis=1)
    exp_val = tl.exp(acc - max_val[:, None])
    sum_val = tl.sum(exp_val, axis=1, keepdim=True)
    softmax_output = exp_val / sum_val
    
    # Multiply with V and store output
    v_start = (pid_m * BLOCK_SIZE_M * heads + pid_n) * seq_len2
    for h in range(heads):
        v_offset = v_start + h * seq_len2
        v_val = tl.load(v_ptr + v_offset + tl.arange(0, BLOCK_SIZE_N), mask=tl.arange(0, BLOCK_SIZE_N) < seq_len2)
        output_val = tl.sum(softmax_output * v_val, axis=1)
        output_offset = ((pid_m * BLOCK_SIZE_M * heads + h) * seq_len2 + pid_n)
        tl.store(output_ptr + output_offset, output_val, mask=(pid_m * BLOCK_SIZE_M * heads + h) * seq_len2 + pid_n < batch_size * heads * seq_len2)


@torch.fx.wrap
def fused_attention(q, k, v, bias, residual):
    batch_size, seq_len, heads, head_dim = q.shape
    _, seq_len2, heads2, _, _ = bias.shape
    
    # Output shape
    output_shape = (batch_size, heads, seq_len2, head_dim)
    output = torch.empty(output_shape, dtype=q.dtype, device=q.device)
    
    # Block sizes for GEMM
    BLOCK_SIZE_M = 32
    BLOCK_SIZE_N = 32
    BLOCK_SIZE_K = 32
    
    # Grid size
    M = batch_size * seq_len / BLOCK_SIZE_M
    N = seq_len2 * heads / BLOCK_SIZE_N
    
    fused_attention_kernel[(M, N), (
        BLOCK_SIZE_M,
        BLOCK_SIZE_N,
        BLOCK_SIZE_K
    )](
        q_ptr=q,
        k_ptr=k,
        v_ptr=v,
        bias_ptr=bias,
        residual_ptr=residual,
        output_ptr=output,
        batch_size=batch_size,
        seq_len=seq_len,
        heads=heads,
        head_dim=head_dim,
        seq_len2=seq_len2,
        heads2=heads2
    )
    
    return output


def replacement_func():
    return fused_attention