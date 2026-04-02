import torch
import triton
import triton.language as tl
import math

# Register multi_head_attention_forward as an FX leaf node via internal mechanism
# so the pattern traces it as a single call_function node instead of expanding it.
try:
    import torch.fx._symbolic_trace as _fxt
    if hasattr(_fxt, '_autowrap_function_ids'):
        _fxt._autowrap_function_ids.add(
            id(torch.nn.functional.multi_head_attention_forward)
        )
    if hasattr(_fxt, '_wrapped_fns_to_patch'):
        _fxt._wrapped_fns_to_patch.setdefault(
            torch.nn.functional.multi_head_attention_forward.__module__, []
        ).append(torch.nn.functional.multi_head_attention_forward.__name__)
except Exception:
    pass


# ============================================================================
# Kernel 1: Tiled GEMM + bias   out[M,N] = X[M,K] @ W[N,K].T + B[N]
# ============================================================================
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 16, 'BLOCK_N':  64, 'BLOCK_K': 64}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N':  64, 'BLOCK_K': 64}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N':  64, 'BLOCK_K': 32}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=8, num_stages=4),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def gemm_bias_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    M, N, K,
    stride_xm, stride_xk,
    stride_wn, stride_wk,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    m_offs = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    n_offs = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        k_offs = k * BLOCK_K + tl.arange(0, BLOCK_K)
        x = tl.load(
            x_ptr + m_offs[:, None] * stride_xm + k_offs[None, :] * stride_xk,
            mask=(m_offs[:, None] < M) & (k_offs[None, :] < K), other=0.0,
        )
        wt = tl.load(
            w_ptr + n_offs[None, :] * stride_wn + k_offs[:, None] * stride_wk,
            mask=(k_offs[:, None] < K) & (n_offs[None, :] < N), other=0.0,
        )
        acc += tl.dot(x, wt)
    bias = tl.load(b_ptr + n_offs, mask=n_offs < N, other=0.0).to(tl.float32)
    acc += bias[None, :]
    out_ptrs = out_ptr + m_offs[:, None] * N + n_offs[None, :]
    tl.store(out_ptrs, acc.to(x_ptr.dtype.element_ty),
             (m_offs[:, None] < M) & (n_offs[None, :] < N))


# ============================================================================
# Kernel 2: Flash Attention forward (online softmax, no dropout)
# ============================================================================
@triton.jit
def flash_attn_fwd_kernel(
    Q_ptr, K_ptr, V_ptr, Out_ptr,
    H, N, D,
    stride_qh, stride_qn, stride_qd,
    stride_kh, stride_kn, stride_kd,
    stride_vh, stride_vn, stride_vd,
    stride_oh, stride_on, stride_od,
    scale,
    BQ: tl.constexpr,
    BK: tl.constexpr,
    BD: tl.constexpr,
):
    pid_h = tl.program_id(0)
    pid_q = tl.program_id(1)
    q_offs = pid_q * BQ + tl.arange(0, BQ)
    d_offs = tl.arange(0, BD)
    q_mask = q_offs < N
    Q = tl.load(
        Q_ptr + pid_h * stride_qh + q_offs[:, None] * stride_qn + d_offs[None, :] * stride_qd,
        mask=q_mask[:, None], other=0.0,
    ).to(tl.float32)
    m   = tl.full((BQ,), float('-inf'), dtype=tl.float32)
    l   = tl.zeros((BQ,), dtype=tl.float32)
    acc = tl.zeros((BQ, BD), dtype=tl.float32)
    for j in range(0, tl.cdiv(N, BK)):
        k_offs  = j * BK + tl.arange(0, BK)
        kv_mask = k_offs < N
        K_blk = tl.load(
            K_ptr + pid_h * stride_kh + k_offs[:, None] * stride_kn + d_offs[None, :] * stride_kd,
            mask=kv_mask[:, None], other=0.0,
        ).to(tl.float32)
        V_blk = tl.load(
            V_ptr + pid_h * stride_vh + k_offs[:, None] * stride_vn + d_offs[None, :] * stride_vd,
            mask=kv_mask[:, None], other=0.0,
        ).to(tl.float32)
        S     = tl.dot(Q, tl.trans(K_blk)) * scale
        S     = tl.where(kv_mask[None, :], S, float('-inf'))
        m_j   = tl.max(S, axis=1)
        m_new = tl.maximum(m, m_j)
        alpha = tl.exp(m - m_new)
        beta  = tl.exp(S - m_new[:, None])
        l     = alpha * l + tl.sum(beta, axis=1)
        acc   = alpha[:, None] * acc + tl.dot(beta, V_blk)
        m     = m_new
    out = acc / l[:, None]
    tl.store(
        Out_ptr + pid_h * stride_oh + q_offs[:, None] * stride_on + d_offs[None, :] * stride_od,
        out.to(Q_ptr.dtype.element_ty),
        mask=q_mask[:, None],
    )


# ============================================================================
# Pattern: MHA (self-attention) + getitem[0]
# ============================================================================
def pattern(in_0, in_1, in_2, in_3, in_4):
    result = torch.nn.functional.multi_head_attention_forward(
        in_4, in_4, in_4, 512, 8, in_3, in_2, None, None, False, 0.0, in_1, in_0,
        training=False, key_padding_mask=None, need_weights=True, attn_mask=None,
        average_attn_weights=True, is_causal=False
    )
    tmp_5 = result[0]
    return tmp_5


def replacement_args(in_0, in_1, in_2, in_3, in_4):
    return (in_0, in_1, in_2, in_3, in_4)


# ============================================================================
# Replacement: Full Triton-based MHA (Flash Attention)
# ============================================================================
@torch.fx.wrap
def optimized_mha(in_0, in_1, in_2, in_3, in_4):
    """
    in_0: [512]        out_proj bias
    in_1: [512, 512]   out_proj weight
    in_2: [1536]       in_proj  bias
    in_3: [1536, 512]  in_proj  weight
    in_4: [150, 1, 512] input (Q=K=V)
    """
    seq_len, bsz, embed_dim = in_4.shape   # 150, 1, 512
    num_heads = 8
    head_dim  = embed_dim // num_heads      # 64
    HN        = bsz * num_heads             # 8
    M         = seq_len * bsz              # 150
    scale     = 1.0 / math.sqrt(head_dim)

    # 1. QKV projection: [M, 512] @ [1536, 512].T + [1536] -> [M, 1536]
    x     = in_4.reshape(M, embed_dim)
    N_qkv = in_3.shape[0]
    K_qkv = in_3.shape[1]
    qkv   = torch.empty(M, N_qkv, device=x.device, dtype=x.dtype)
    gemm_bias_kernel[
        lambda META: (triton.cdiv(M, META['BLOCK_M']),
                      triton.cdiv(N_qkv, META['BLOCK_N']))
    ](
        x, in_3, in_2, qkv,
        M, N_qkv, K_qkv,
        x.stride(0), x.stride(1),
        in_3.stride(0), in_3.stride(1),
    )

    # 2. Split & reshape -> [HN, seq_len, head_dim]
    q_2d, k_2d, v_2d = qkv.chunk(3, dim=-1)
    q = q_2d.view(seq_len, HN, head_dim).permute(1, 0, 2).contiguous()
    k = k_2d.view(seq_len, HN, head_dim).permute(1, 0, 2).contiguous()
    v = v_2d.view(seq_len, HN, head_dim).permute(1, 0, 2).contiguous()

    # 3. Flash Attention
    BQ, BK_fa, BD = 32, 32, head_dim
    attn_out = torch.empty_like(q)
    flash_attn_fwd_kernel[
        (HN, triton.cdiv(seq_len, BQ))
    ](
        q, k, v, attn_out,
        HN, seq_len, head_dim,
        q.stride(0), q.stride(1), q.stride(2),
        k.stride(0), k.stride(1), k.stride(2),
        v.stride(0), v.stride(1), v.stride(2),
        attn_out.stride(0), attn_out.stride(1), attn_out.stride(2),
        scale,
        BQ=BQ, BK=BK_fa, BD=BD,
    )

    # 4. Merge heads: [HN, seq_len, head_dim] -> [M, embed_dim]
    attn_2d = attn_out.permute(1, 0, 2).contiguous().view(M, embed_dim)

    # 5. Output projection: [M, 512] @ [512, 512].T + [512] -> [M, 512]
    N_out = in_1.shape[0]
    K_out = in_1.shape[1]
    out   = torch.empty(M, N_out, device=attn_2d.device, dtype=attn_2d.dtype)
    gemm_bias_kernel[
        lambda META: (triton.cdiv(M, META['BLOCK_M']),
                      triton.cdiv(N_out, META['BLOCK_N']))
    ](
        attn_2d, in_1, in_0, out,
        M, N_out, K_out,
        attn_2d.stride(0), attn_2d.stride(1),
        in_1.stride(0), in_1.stride(1),
    )

    return out.view(seq_len, bsz, embed_dim)


def replacement_func():
    return optimized_mha