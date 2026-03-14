import torch
import triton
import triton.language as tl

def pattern(in_0, in_1):
    """Match the attention computation pattern with proper operations"""
    # Match exact operations from model.py
    tmp_0 = in_0 * 1.0
    tmp_1 = torch.nn.functional.softmax(tmp_0, dim=-1, dtype=torch.float32)
    tmp_2 = tmp_1.to(torch.float32)
    tmp_3 = torch.nn.functional.dropout(tmp_2, p=0.0, training=False)
    tmp_4 = torch.matmul(tmp_3, in_1)
    tmp_5 = tmp_4.transpose(1, 2)
    tmp_6 = tmp_5.contiguous()
    tmp_7 = tmp_6.reshape(1, 257, -1)
    tmp_8 = tmp_7.contiguous()
    return (tmp_8,)

def replacement_args(in_0, in_1):
    return (in_0, in_1)

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32}, num_stages=2, num_warps=8),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def fused_attention_kernel(
    # Pointers
    attn_ptr, value_ptr, out_ptr,
    # Shapes
    batch, num_heads, seq_len, head_dim,
    M, N, K,
    # Strides for attn [1, 16, 257, 257]
    stride_attn_b, stride_attn_h, stride_attn_m, stride_attn_k,
    # Strides for value [1, 16, 257, 80]
    stride_val_b, stride_val_h, stride_val_k, stride_val_n,
    # Strides for output [1, 257, 16, 80]
    stride_out_b, stride_out_m, stride_out_h, stride_out_n,
    # Block sizes
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    """
    Fused kernel for: softmax -> matmul -> transpose
    Input: attn [batch, num_heads, M, K], value [batch, num_heads, K, N]
    Output: [batch, M, num_heads, N]
    """
    # Program IDs
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_m = tl.program_id(2)
    pid_n = tl.program_id(3)
    
    # Offsets
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    
    # Masks
    mask_m = offs_m < M
    mask_n = offs_n < N
    
    # Pointers to attention row (need full row for softmax)
    attn_row_ptr = attn_ptr + pid_b * stride_attn_b + pid_h * stride_attn_h
    
    # For each row in this block, compute softmax and matmul
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    for m_idx in range(BLOCK_SIZE_M):
        m = pid_m * BLOCK_SIZE_M + m_idx
        if m < M:
            # Load attention row for softmax
            attn_offs = m * stride_attn_m + tl.arange(0, K)
            attn_mask = attn_offs < K * stride_attn_k
            attn_row = tl.load(attn_row_ptr + attn_offs, mask=(m < M), other=-float('inf'))
            
            # Softmax: subtract max for numerical stability
            row_max = tl.max(attn_row, axis=0)
            attn_row = attn_row - row_max
            attn_row = tl.exp(attn_row)
            row_sum = tl.sum(attn_row, axis=0)
            attn_row = attn_row / row_sum
            
            # Matmul: compute dot product with value columns
            val_ptr = value_ptr + pid_b * stride_val_b + pid_h * stride_val_h
            
            for k in range(0, K, BLOCK_SIZE_K):
                k_offs = k + offs_k
                k_mask = k_offs < K
                
                # Load attention weights for this k block
                a = tl.load(attn_row_ptr + m * stride_attn_m + k_offs * stride_attn_k, 
                           mask=k_mask & (m < M), other=0.0)
                
                # Load value block and compute
                for n_idx in range(BLOCK_SIZE_N):
                    n = pid_n * BLOCK_SIZE_N + n_idx
                    if n < N:
                        v = tl.load(val_ptr + k_offs[:, None] * stride_val_k + n * stride_val_n, 
                                   mask=k_mask[:, None], other=0.0)
                        acc[m_idx, n_idx] += tl.sum(a * v[:, 0])
    
    # Store to output with transposed layout [batch, M, num_heads, N]
    offs_out_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_out_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    
    out_ptrs = (out_ptr + pid_b * stride_out_b + 
                offs_out_m[:, None] * stride_out_m + 
                pid_h * stride_out_h + 
                offs_out_n[None, :] * stride_out_n)
    
    mask = (offs_out_m[:, None] < M) & (offs_out_n[None, :] < N)
    tl.store(out_ptrs, acc, mask=mask)


@torch.fx.wrap
def fused_attention_optimized(attn, value):
    """
    Optimized attention: softmax(attn) @ value with transpose
    attn: [batch, num_heads, seq_len, seq_len]
    value: [batch, num_heads, seq_len, head_dim]
    output: [batch, seq_len, num_heads * head_dim]
    """
    batch, num_heads, seq_len, _ = attn.shape
    _, _, _, head_dim = value.shape
    
    M = seq_len
    N = head_dim
    K = seq_len
    
    # Output shape: [batch, seq_len, num_heads, head_dim]
    output = torch.empty((batch, M, num_heads, N), device=attn.device, dtype=torch.float32)
    
    # Grid: (batch, num_heads, M_blocks, N_blocks)
    BLOCK_SIZE_M = 32
    BLOCK_SIZE_N = 64
    BLOCK_SIZE_K = 32
    
    grid = (
        batch,
        num_heads,
        triton.cdiv(M, BLOCK_SIZE_M),
        triton.cdiv(N, BLOCK_SIZE_N),
    )
    
    fused_attention_kernel[grid](
        attn, value, output,
        batch, num_heads, seq_len, head_dim,
        M, N, K,
        attn.stride(0), attn.stride(1), attn.stride(2), attn.stride(3),
        value.stride(0), value.stride(1), value.stride(2), value.stride(3),
        output.stride(0), output.stride(1), output.stride(2), output.stride(3),
    )
    
    # Reshape to [batch, seq_len, num_heads * head_dim]
    output = output.reshape(batch, M, -1)
    
    return output


def replacement_func():
    return fused_attention_optimized