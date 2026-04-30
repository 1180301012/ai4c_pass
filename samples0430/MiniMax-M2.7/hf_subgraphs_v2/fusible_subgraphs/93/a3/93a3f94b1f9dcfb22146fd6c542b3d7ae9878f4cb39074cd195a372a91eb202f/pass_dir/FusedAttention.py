import torch
import triton
import triton.language as tl

@triton.jit
def fused_attention_kernel(
    Q_ptr, K_ptr, V_ptr,
    output_ptr,
    stride_qb, stride_qh, stride_qm, stride_qk,
    stride_kb, stride_kh, stride_kn, stride_kk,
    stride_vb, stride_vh, stride_vn, stride_vk,
    stride_ob, stride_oh, stride_om, stride_ok,
    batch_size, num_heads, seq_len, head_dim,
    scale: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    # Program ID
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_m = tl.program_id(2)
    
    # Offsets
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    
    # Stride offsets
    q_offset = (pid_b * stride_qb + pid_h * stride_qh + offs_m[:, None] * stride_qm + offs_k[None, :] * stride_qk)
    k_offset = (pid_b * stride_kb + pid_h * stride_kh + offs_n[:, None] * stride_kn + offs_k[None, :] * stride_kk)
    v_offset = (pid_b * stride_vb + pid_h * stride_vh + offs_n[:, None] * stride_vn + offs_k[None, :] * stride_vk)
    
    # Load Q: [BLOCK_SIZE_M, head_dim]
    q = tl.load(Q_ptr + q_offset, mask=(offs_m[:, None] < seq_len) & (offs_k[None, :] < head_dim), other=0.0)
    
    # Compute Q @ K^T in blocks
    # acc shape: [BLOCK_SIZE_M, BLOCK_SIZE_N]
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    # Blocked matmul for attention scores
    for k in range(0, (head_dim + BLOCK_SIZE_K - 1) // BLOCK_SIZE_K):
        k_offset_k = (pid_b * stride_kb + pid_h * stride_kh + 
                      offs_n[:, None] * stride_kn + 
                      (offs_k + k * BLOCK_SIZE_K)[None, :] * stride_kk)
        k_mask = (offs_n[:, None] < seq_len) & ((offs_k + k * BLOCK_SIZE_K)[None, :] < head_dim)
        k_block = tl.load(K_ptr + k_offset_k, mask=k_mask, other=0.0)
        
        q_offset_k = (pid_b * stride_qb + pid_h * stride_qh + 
                      offs_m[:, None] * stride_qm + 
                      (offs_k + k * BLOCK_SIZE_K)[None, :] * stride_qk)
        q_mask = (offs_m[:, None] < seq_len) & ((offs_k + k * BLOCK_SIZE_K)[None, :] < head_dim)
        q_block = tl.load(Q_ptr + q_offset_k, mask=q_mask, other=0.0)
        
        acc += tl.dot(tl.to_tlfloat32(q_block), tl.trans(tl.to_tlfloat32(k_block)))
    
    # Scale attention scores
    acc = acc * scale
    
    # Softmax along N dimension (seq_len)
    # Exponentiate and sum for normalization
    acc_minus_max = acc - tl.max(acc, axis=1, keepdims=True)
    acc_exp = tl.exp(acc_minus_max)
    acc_exp_sum = tl.sum(acc_exp, axis=1, keepdims=True) + 1e-12
    acc_softmax = acc_exp / acc_exp_sum
    
    # Load V: [seq_len, head_dim]
    # Compute final output: softmax_scores @ V
    # output = [BLOCK_SIZE_M, head_dim]
    output = tl.zeros((BLOCK_SIZE_M, head_dim), dtype=tl.float32)
    
    for n_block in range(0, (seq_len + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N):
        n_offset = (pid_b * stride_vb + pid_h * stride_vh + 
                    (offs_n + n_block * BLOCK_SIZE_N)[:, None] * stride_vn + 
                    offs_k[None, :] * stride_vk)
        n_mask = ((offs_n + n_block * BLOCK_SIZE_N)[:, None] < seq_len) & (offs_k[None, :] < head_dim)
        v_block = tl.load(V_ptr + n_offset, mask=n_mask, other=0.0)
        
        output += tl.dot(tl.trans(acc_softmax[:, n_block * BLOCK_SIZE_N:(n_block + 1) * BLOCK_SIZE_N].reshape(BLOCK_SIZE_M, BLOCK_SIZE_N)), 
                         tl.to_tlfloat32(v_block))
    
    # Store output
    o_offset = (pid_b * stride_ob + pid_h * stride_oh + 
                offs_m[:, None] * stride_om + offs_k[None, :] * stride_ok)
    o_mask = (offs_m[:, None] < seq_len) & (offs_k[None, :] < head_dim)
    tl.store(output_ptr + o_offset, output, mask=o_mask)


@triton.jit
def fused_attention_kernel_v2(
    Q_ptr, K_ptr, V_ptr,
    output_ptr,
    stride_qb, stride_qh, stride_qm, stride_qk,
    stride_kb, stride_kh, stride_kn, stride_kk,
    stride_vb, stride_vh, stride_vn, stride_vk,
    stride_ob, stride_oh, stride_om, stride_ok,
    batch_size, num_heads, seq_len, head_dim,
    scale: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    # Program ID
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_m = tl.program_id(2)
    
    # Offsets
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, head_dim)
    
    # Stride offsets for Q: [B, H, M, K]
    q_offset = (pid_b * stride_qb + pid_h * stride_qh + 
                offs_m[:, None] * stride_qm + offs_k[None, :] * stride_qk)
    q_mask = (offs_m[:, None] < seq_len) & (offs_k[None, :] < head_dim)
    
    # Load Q block
    q = tl.load(Q_ptr + q_offset, mask=q_mask, other=0.0)
    q = tl.to_tlfloat32(q)
    
    # Compute attention scores for this Q block against all K
    # acc shape: [BLOCK_SIZE_M, seq_len]
    acc = tl.zeros((BLOCK_SIZE_M, seq_len), dtype=tl.float32)
    
    # Process K in blocks
    num_k_blocks = (seq_len + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N
    for kb in range(num_k_blocks):
        offs_n_k = kb * BLOCK_SIZE_N + offs_n
        k_offset = (pid_b * stride_kb + pid_h * stride_kh + 
                    offs_n_k[:, None] * stride_kn + offs_k[None, :] * stride_kk)
        k_mask = (offs_n_k[:, None] < seq_len) & (offs_k[None, :] < head_dim)
        
        k = tl.load(K_ptr + k_offset, mask=k_mask, other=0.0)
        k = tl.to_tlfloat32(k)
        
        # Q @ K^T for this block
        block_scores = tl.dot(q, tl.trans(k))  # [BLOCK_SIZE_M, BLOCK_SIZE_N]
        
        # Store to acc at the right position
        store_offs_n = kb * BLOCK_SIZE_N + offs_n
        store_mask = (offs_m[:, None] < seq_len) & (store_offs_n[None, :] < seq_len)
        tl.store(acc + offs_m[:, None] * seq_len + store_offs_n[None, :], 
                 block_scores, mask=store_mask)
    
    # Softmax
    acc_max = tl.max(acc, axis=1, keepdims=True)
    acc = acc - acc_max
    acc_exp = tl.exp(acc)
    acc_sum = tl.sum(acc_exp, axis=1, keepdims=True) + 1e-12
    acc = acc_exp / acc_sum
    
    # Compute output: softmax_scores @ V
    output = tl.zeros((BLOCK_SIZE_M, head_dim), dtype=tl.float32)
    
    for vb in range(num_k_blocks):
        offs_n_v = vb * BLOCK_SIZE_N + offs_n
        v_offset = (pid_b * stride_vb + pid_h * stride_vh + 
                    offs_n_v[:, None] * stride_vn + offs_k[None, :] * stride_vk)
        v_mask = (offs_n_v[:, None] < seq_len) & (offs_k[None, :] < head_dim)
        
        v = tl.load(V_ptr + v_offset, mask=v_mask, other=0.0)
        v = tl.to_tlfloat32(v)
        
        # Get attention scores for this V block
        scores_block = tl.load(acc + offs_m[:, None] * seq_len + (offs_n_v)[None, :],
                               mask=(offs_m[:, None] < seq_len) & ((offs_n_v)[None, :] < seq_len),
                               other=0.0)
        
        output += tl.dot(scores_block, v)
    
    # Scale output (optional, for numerical stability)
    output = output  # Already in float32
    
    # Store output: [B, H, M, K]
    o_offset = (pid_b * stride_ob + pid_h * stride_oh + 
                offs_m[:, None] * stride_om + offs_k[None, :] * stride_ok)
    o_mask = (offs_m[:, None] < seq_len) & (offs_k[None, :] < head_dim)
    tl.store(output_ptr + o_offset, output, mask=o_mask)


# Optimized kernel with autotuning
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 256}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 128}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 32}, num_stages=3, num_warps=4),
    ],
    key=['seq_len', 'head_dim'],
)
@triton.jit
def fused_attention_autotuned(
    Q_ptr, K_ptr, V_ptr,
    output_ptr,
    stride_qb, stride_qh, stride_qm, stride_qk,
    stride_kb, stride_kh, stride_kn, stride_kk,
    stride_vb, stride_vh, stride_vn, stride_vk,
    stride_ob, stride_oh, stride_om, stride_ok,
    batch_size, num_heads, seq_len, head_dim,
    scale: tl.constexpr,
):
    BLOCK_SIZE_M = tl.constexpr(64)
    BLOCK_SIZE_N = tl.constexpr(64)
    
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_m = tl.program_id(2)
    
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, head_dim)
    
    # Load Q
    q_offset = (pid_b * stride_qb + pid_h * stride_qh + 
                offs_m[:, None] * stride_qm + offs_k[None, :] * stride_qk)
    q_mask = (offs_m[:, None] < seq_len) & (offs_k[None, :] < head_dim)
    q = tl.load(Q_ptr + q_offset, mask=q_mask, other=0.0)
    q = tl.to_tlfloat32(q)
    
    # Compute attention scores (Q @ K^T)
    num_k_blocks = (seq_len + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    for kb in range(num_k_blocks):
        offs_n_k = kb * BLOCK_SIZE_N + offs_n
        
        k_offset = (pid_b * stride_kb + pid_h * stride_kh + 
                    offs_n_k[:, None] * stride_kn + offs_k[None, :] * stride_kk)
        k_mask = (offs_n_k[:, None] < seq_len) & (offs_k[None, :] < head_dim)
        
        k = tl.load(K_ptr + k_offset, mask=k_mask, other=0.0)
        k = tl.to_tlfloat32(k)
        
        acc += tl.dot(q, tl.trans(k))
    
    # Scale
    acc = acc * scale
    
    # Softmax
    acc_max = tl.max(acc, axis=1, keepdims=True)
    acc_minus_max = acc - acc_max
    exp_vals = tl.exp(acc_minus_max)
    exp_sum = tl.sum(exp_vals, axis=1, keepdims=True) + 1e-12
    attn_weights = exp_vals / exp_sum
    
    # Compute output (attn_weights @ V)
    output = tl.zeros((BLOCK_SIZE_M, head_dim), dtype=tl.float32)
    
    for vb in range(num_k_blocks):
        offs_n_v = vb * BLOCK_SIZE_N + offs_n
        
        v_offset = (pid_b * stride_vb + pid_h * stride_vh + 
                    offs_n_v[:, None] * stride_vn + offs_k[None, :] * stride_vk)
        v_mask = (offs_n_v[:, None] < seq_len) & (offs_k[None, :] < head_dim)
        
        v = tl.load(V_ptr + v_offset, mask=v_mask, other=0.0)
        v = tl.to_tlfloat32(v)
        
        output += tl.dot(attn_weights, v)
    
    # Store output
    o_offset = (pid_b * stride_ob + pid_h * stride_oh + 
                offs_m[:, None] * stride_om + offs_k[None, :] * stride_ok)
    o_mask = (offs_m[:, None] < seq_len) & (offs_k[None, :] < head_dim)
    tl.store(output_ptr + o_offset, output, mask=o_mask)


# Kernel wrapper functions for different dtypes
@torch.fx.wrap
def triton_fused_attention_bfloat16(q, k, v):
    """Fused attention kernel for bfloat16 inputs"""
    B, H, M, K = q.shape
    _, _, N, _ = k.shape
    scale = 1.0 / (K ** 0.5)
    
    # Grid: (batch, heads, M_blocks)
    BLOCK_SIZE_M = 32
    grid = (B, H, (M + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M)
    
    # Allocate output
    output = torch.empty((B, H, M, K), device=q.device, dtype=torch.float32)
    
    fused_attention_autotuned[grid](
        q, k, v, output,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        output.stride(0), output.stride(1), output.stride(2), output.stride(3),
        B, H, M, K,
        scale,
        num_stages=3,
        num_warps=8,
    )
    
    return output


@torch.fx.wrap
def triton_fused_attention_float16(q, k, v):
    """Fused attention kernel for float16 inputs"""
    B, H, M, K = q.shape
    _, _, N, _ = k.shape
    scale = 1.0 / (K ** 0.5)
    
    BLOCK_SIZE_M = 32
    grid = (B, H, (M + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M)
    
    output = torch.empty((B, H, M, K), device=q.device, dtype=torch.float32)
    
    fused_attention_autotuned[grid](
        q, k, v, output,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        output.stride(0), output.stride(1), output.stride(2), output.stride(3),
        B, H, M, K,
        scale,
        num_stages=3,
        num_warps=8,
    )
    
    return output


@torch.fx.wrap
def triton_fused_attention_float32(q, k, v):
    """Fused attention kernel for float32 inputs"""
    B, H, M, K = q.shape
    _, _, N, _ = k.shape
    scale = 1.0 / (K ** 0.5)
    
    BLOCK_SIZE_M = 32
    grid = (B, H, (M + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M)
    
    output = torch.empty((B, H, M, K), device=q.device, dtype=torch.float32)
    
    fused_attention_autotuned[grid](
        q, k, v, output,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        output.stride(0), output.stride(1), output.stride(2), output.stride(3),
        B, H, M, K,
        scale,
        num_stages=3,
        num_warps=8,
    )
    
    return output


def pattern(in_0, in_1, in_2):
    """
    Match the core attention computation pattern:
    matmul -> scale -> softmax -> matmul (for bfloat16)
    """
    # First matmul: Q @ K^T
    matmul = torch.matmul(in_0, in_1)
    # Scale by 1.0
    tmp_1 = matmul * 1.0
    # Softmax
    tmp_2 = torch.nn.functional.softmax(tmp_1, dim=-1, dtype=torch.float32)
    # Type conversion
    tmp_3 = tmp_2.to(torch.float32)
    # Dropout with p=0.0
    tmp_4 = torch.nn.functional.dropout(tmp_3, p=0.0, training=False)
    # Type conversion to bfloat16
    to = tmp_4.to(torch.bfloat16)
    # Second matmul
    matmul_1 = torch.matmul(to, in_2)
    return matmul_1


def replacement_args(in_0, in_1, in_2):
    return (in_0, in_1, in_2, "bfloat16")


# Module-level dispatch wrapper function
def fused_attention_dispatch(q, k, v, route):
    if route == "bfloat16":
        output = triton_fused_attention_bfloat16(q, k, v)
    elif route == "float16":
        output = triton_fused_attention_float16(q, k, v)
    elif route == "float32":
        output = triton_fused_attention_float32(q, k, v)
    else:
        raise ValueError(f"Unknown route: {route}")
    
    return output


def replacement_func():
    return fused_attention_dispatch