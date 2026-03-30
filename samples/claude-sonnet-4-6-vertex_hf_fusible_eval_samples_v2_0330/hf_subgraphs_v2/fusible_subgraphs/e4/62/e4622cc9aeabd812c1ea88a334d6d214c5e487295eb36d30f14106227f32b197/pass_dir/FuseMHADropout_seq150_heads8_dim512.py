"""
Pass: FuseMHADropout_seq150_heads8_dim512

Pattern: MHA(self-attn, need_weights=True) + getitem[0] + dropout(p=0) + dropout(p=0)
Replace: Triton Flash-Attention + Triton linear projections
         (avoids attention-weight computation and two no-op dropout launches)
"""
import torch
import triton
import triton.language as tl
import math

# ─── Triton kernel: fused matmul + bias-add ────────────────────────────────────
# Computes:  Out[m, n] = sum_k(A[m,k] * B[k,n]) + bias[n]
# A : [M, K]  (row-major, contiguous)
# B : [N, K]  (row-major → we treat it as K×N by iterating over K)
# Actually we pass B transposed: Bt[K, N] stored as [N, K] → we do A @ B.T
# Simpler: pass W (shape [N, K]) and compute A @ W.T + bias → shape [M, N]

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64,  'BLOCK_K': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64,  'BLOCK_K': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 64,  'BLOCK_K': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=4, num_stages=3),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def _linear_kernel(
    A_ptr, W_ptr, bias_ptr, Out_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_wn, stride_wk,
    stride_om, stride_on,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """Compute Out = A @ W.T + bias"""
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    m_start = pid_m * BLOCK_M
    n_start = pid_n * BLOCK_N

    m_offs = m_start + tl.arange(0, BLOCK_M)
    n_offs = n_start + tl.arange(0, BLOCK_N)
    k_offs = tl.arange(0, BLOCK_K)

    acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)

    for k_start in range(0, K, BLOCK_K):
        k_o = k_start + k_offs
        k_mask = k_o < K

        # Load A tile: [BLOCK_M, BLOCK_K]
        a_ptrs = A_ptr + m_offs[:, None] * stride_am + k_o[None, :] * stride_ak
        a_mask = (m_offs[:, None] < M) & k_mask[None, :]
        a = tl.load(a_ptrs, mask=a_mask, other=0.0).to(tl.float32)

        # Load W tile: [BLOCK_N, BLOCK_K]  (W is weight [N, K])
        w_ptrs = W_ptr + n_offs[:, None] * stride_wn + k_o[None, :] * stride_wk
        w_mask = (n_offs[:, None] < N) & k_mask[None, :]
        w = tl.load(w_ptrs, mask=w_mask, other=0.0).to(tl.float32)

        acc += tl.dot(a, tl.trans(w))

    # Add bias
    b_mask = n_offs < N
    bias = tl.load(bias_ptr + n_offs, mask=b_mask, other=0.0).to(tl.float32)
    acc += bias[None, :]

    # Store
    out_mask = (m_offs[:, None] < M) & (n_offs[None, :] < N)
    tl.store(Out_ptr + m_offs[:, None] * stride_om + n_offs[None, :] * stride_on,
             acc, mask=out_mask)


def triton_linear(a, w, bias):
    """a: [M, K], w: [N, K], bias: [N]  →  [M, N]"""
    M, K = a.shape
    N    = w.shape[0]
    out  = torch.empty((M, N), device=a.device, dtype=torch.float32)
    grid = (triton.cdiv(M, 32), triton.cdiv(N, 64))
    _linear_kernel[grid](
        a, w, bias, out,
        M, N, K,
        a.stride(0), a.stride(1),
        w.stride(0), w.stride(1),
        out.stride(0), out.stride(1),
    )
    return out


# ─── Triton Flash Attention Kernel ────────────────────────────────────────────

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 16}, num_warps=4, num_stages=2),
    ],
    key=['seq_len'],
)
@triton.jit
def _flash_attn_fwd(
    Q_ptr, K_ptr, V_ptr, Out_ptr,
    stride_h, stride_s, stride_d,
    num_heads_x_bsz, seq_len,
    scale,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    HEAD_DIM: tl.constexpr,
):
    head_idx = tl.program_id(0)
    m_block  = tl.program_id(1)

    m_start = m_block * BLOCK_M
    m_offs  = m_start + tl.arange(0, BLOCK_M)
    d_offs  = tl.arange(0, HEAD_DIM)

    q_base = head_idx * stride_h
    q_ptrs = Q_ptr + q_base + m_offs[:, None] * stride_s + d_offs[None, :] * stride_d
    q_mask = m_offs[:, None] < seq_len
    q      = tl.load(q_ptrs, mask=q_mask, other=0.0).to(tl.float32) * scale

    m_i = tl.full([BLOCK_M], float('-inf'), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)

    for n_start in range(0, seq_len, BLOCK_N):
        n_offs = n_start + tl.arange(0, BLOCK_N)
        n_mask = n_offs < seq_len

        k_ptrs = K_ptr + head_idx * stride_h + n_offs[:, None] * stride_s + d_offs[None, :] * stride_d
        k      = tl.load(k_ptrs, mask=n_mask[:, None], other=0.0).to(tl.float32)

        scores = tl.dot(q, tl.trans(k))
        scores = tl.where(n_mask[None, :], scores, float('-inf'))

        m_ij  = tl.max(scores, axis=1)
        m_new = tl.maximum(m_i, m_ij)
        alpha = tl.exp(m_i - m_new)
        p     = tl.exp(scores - m_new[:, None])

        l_i = l_i * alpha + tl.sum(p, axis=1)
        acc = acc * alpha[:, None]

        v_ptrs = V_ptr + head_idx * stride_h + n_offs[:, None] * stride_s + d_offs[None, :] * stride_d
        v      = tl.load(v_ptrs, mask=n_mask[:, None], other=0.0).to(tl.float32)

        acc = acc + tl.dot(p, v)
        m_i = m_new

    acc = acc / l_i[:, None]

    out_ptrs = Out_ptr + head_idx * stride_h + m_offs[:, None] * stride_s + d_offs[None, :] * stride_d
    out_mask = m_offs[:, None] < seq_len
    tl.store(out_ptrs, acc, mask=out_mask)


# ─── Python wrapper ────────────────────────────────────────────────────────────

@torch.fx.wrap
def optimized_mha_fwd(in_4, in_3, in_2, in_1, in_0):
    """
    Fused replacement for MHA (need_weights=True) + getitem[0] + 2x identity dropout.

    in_4 : [seq_len, bsz, embed_dim]  – query = key = value
    in_3 : [3*embed_dim, embed_dim]   – in_proj_weight
    in_2 : [3*embed_dim]              – in_proj_bias
    in_1 : [embed_dim, embed_dim]     – out_proj_weight
    in_0 : [embed_dim]                – out_proj_bias
    """
    seq_len, bsz, embed_dim = in_4.shape
    num_heads = 8
    head_dim  = embed_dim // num_heads   # 64

    # ── 1. QKV projection via Triton linear ─────────────────────────────────
    x   = in_4.reshape(seq_len * bsz, embed_dim).contiguous()  # [150, 512]
    # in_3: [1536, 512],  in_2: [1536]
    # cast inputs to float32 for Triton kernel
    x_f32   = x.to(torch.float32)
    w3_f32  = in_3.to(torch.float32)
    b2_f32  = in_2.to(torch.float32)
    qkv     = triton_linear(x_f32, w3_f32, b2_f32)  # [150, 1536] float32

    q = qkv[:, :embed_dim].contiguous()          # [150, 512]
    k = qkv[:, embed_dim:2*embed_dim].contiguous()
    v = qkv[:, 2*embed_dim:].contiguous()

    # ── 2. Reshape to [bsz*num_heads, seq_len, head_dim] ────────────────────
    q = q.reshape(seq_len, bsz * num_heads, head_dim).permute(1, 0, 2).contiguous()
    k = k.reshape(seq_len, bsz * num_heads, head_dim).permute(1, 0, 2).contiguous()
    v = v.reshape(seq_len, bsz * num_heads, head_dim).permute(1, 0, 2).contiguous()

    # ── 3. Triton Flash Attention ────────────────────────────────────────────
    num_heads_x_bsz = bsz * num_heads
    out_attn = torch.empty(num_heads_x_bsz, seq_len, head_dim,
                           device=in_4.device, dtype=torch.float32)
    scale    = 1.0 / math.sqrt(head_dim)

    stride_h = seq_len * head_dim
    stride_s = head_dim
    stride_d = 1

    grid = (num_heads_x_bsz, triton.cdiv(seq_len, 16))
    _flash_attn_fwd[grid](
        q, k, v, out_attn,
        stride_h, stride_s, stride_d,
        num_heads_x_bsz, seq_len,
        scale,
        HEAD_DIM=head_dim,
    )

    # ── 4. Output projection via Triton linear ───────────────────────────────
    # [num_heads*bsz, seq_len, head_dim] → [seq_len*bsz, embed_dim]
    out_flat = out_attn.permute(1, 0, 2).contiguous().reshape(seq_len * bsz, embed_dim)
    w1_f32   = in_1.to(torch.float32)
    b0_f32   = in_0.to(torch.float32)
    output   = triton_linear(out_flat, w1_f32, b0_f32)   # [150, 512] float32

    # Cast back to original dtype and reshape
    output = output.to(in_4.dtype).reshape(seq_len, bsz, embed_dim)
    return output


# ─── Pattern (must mirror model.py exactly) ───────────────────────────────────

def pattern(in_4, in_3, in_2, in_1, in_0):
    result = torch.nn.functional.multi_head_attention_forward(
        in_4, in_4, in_4, 512, 8, in_3, in_2, None, None, False, 0.0,
        in_1, in_0,
        training=False,
        key_padding_mask=None,
        need_weights=True,
        attn_mask=None,
        average_attn_weights=True,
        is_causal=False,
    )
    tmp_5 = result[0]
    tmp_6 = torch.nn.functional.dropout(tmp_5, 0.0, False, False)
    tmp_7 = torch.nn.functional.dropout(tmp_6, 0.0, False, False)
    return tmp_7


# ─── Argument extraction ──────────────────────────────────────────────────────

def replacement_args(in_4, in_3, in_2, in_1, in_0):
    return (in_4, in_3, in_2, in_1, in_0)


# ─── Replacement factory ──────────────────────────────────────────────────────

def replacement_func():
    return optimized_mha_fwd