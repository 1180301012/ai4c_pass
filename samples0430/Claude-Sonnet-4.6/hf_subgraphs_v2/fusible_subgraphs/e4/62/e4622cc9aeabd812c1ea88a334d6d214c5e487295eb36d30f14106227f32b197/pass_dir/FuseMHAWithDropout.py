import torch
import triton
import triton.language as tl
import math


# ---------------------------------------------------------------------------
# Kernel 1: Fused linear (GEMM) + bias  –  computes  out = x @ w^T + b
#   x   : [M, K]  (row-major)
#   w   : [N, K]  (row-major, i.e. each row is one output neuron's weights)
#   b   : [N]
#   out : [M, N]
# ---------------------------------------------------------------------------
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_stages=4, num_warps=4),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def linear_bias_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    M, N, K,
    stride_xm, stride_xk,
    stride_wn, stride_wk,
    stride_om, stride_on,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(tl.cdiv(K, BLOCK_K)):
        offs_k = k * BLOCK_K + tl.arange(0, BLOCK_K)

        # x tile:  [BLOCK_M, BLOCK_K]
        x = tl.load(
            x_ptr + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk,
            mask=(offs_m[:, None] < M) & (offs_k[None, :] < K),
            other=0.0,
        )
        # w tile:  [BLOCK_N, BLOCK_K]  (rows = output neurons, cols = input dims)
        w = tl.load(
            w_ptr + offs_n[:, None] * stride_wn + offs_k[None, :] * stride_wk,
            mask=(offs_n[:, None] < N) & (offs_k[None, :] < K),
            other=0.0,
        )

        # x @ w^T  → accumulate into [BLOCK_M, BLOCK_N]
        acc = tl.dot(x, tl.trans(w), acc)

    # add bias
    b = tl.load(b_ptr + offs_n, mask=offs_n < N, other=0.0)
    acc = acc + b[None, :]

    # store
    mask_out = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(
        out_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on,
        acc.to(out_ptr.dtype.element_ty),
        mask=mask_out,
    )


# ---------------------------------------------------------------------------
# Kernel 2: Flash-Attention forward  (online softmax, no causal mask)
#
#  proj    : [T_B, 3*E]   in-projection output (Q, K, V concatenated)
#              Q[s,h,d] @ proj[s,  h*HEAD_DIM + d]
#              K[s,h,d] @ proj[s,  E + h*HEAD_DIM + d]
#              V[s,h,d] @ proj[s, 2E + h*HEAD_DIM + d]
#  out_ptr : [T_B, E]    attention output (heads interleaved: out[s, h*HD+d])
#  T_B     : total tokens = seq_len * batch
#  scale   : 1 / sqrt(head_dim)
# ---------------------------------------------------------------------------
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 32}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 32}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 32}, num_stages=4, num_warps=8),
    ],
    key=['T_B'],
)
@triton.jit
def flash_attn_kernel(
    proj_ptr,
    out_ptr,
    T_B,
    scale,
    EMBED_DIM: tl.constexpr,   # E = 512
    HEAD_DIM:  tl.constexpr,   # E / H = 64
    BLOCK_M:   tl.constexpr,
    BLOCK_N:   tl.constexpr,
):
    head_id   = tl.program_id(0)   # [0, NUM_HEADS)
    block_m   = tl.program_id(1)   # [0, ceil(T_B / BLOCK_M))

    m_start = block_m * BLOCK_M
    offs_m  = m_start + tl.arange(0, BLOCK_M)
    offs_d  = tl.arange(0, HEAD_DIM)

    # column offsets for Q, K, V within each row of proj
    q_col = head_id * HEAD_DIM
    k_col = EMBED_DIM     + head_id * HEAD_DIM
    v_col = 2 * EMBED_DIM + head_id * HEAD_DIM

    PROJ_STRIDE = 3 * EMBED_DIM   # row stride of proj

    # Load Q block [BLOCK_M, HEAD_DIM] – cast to float32 for stable softmax
    q = tl.load(
        proj_ptr + offs_m[:, None] * PROJ_STRIDE + (q_col + offs_d[None, :]),
        mask=offs_m[:, None] < T_B,
        other=0.0,
    ).to(tl.float32)

    # Running statistics for online softmax
    m_i  = tl.full((BLOCK_M,), float('-inf'), dtype=tl.float32)
    l_i  = tl.zeros((BLOCK_M,), dtype=tl.float32)
    acc  = tl.zeros((BLOCK_M, HEAD_DIM), dtype=tl.float32)

    # Iterate over K / V blocks
    for n_start in range(0, T_B, BLOCK_N):
        offs_n  = n_start + tl.arange(0, BLOCK_N)
        mask_n  = offs_n < T_B

        # K block [BLOCK_N, HEAD_DIM]
        k = tl.load(
            proj_ptr + offs_n[:, None] * PROJ_STRIDE + (k_col + offs_d[None, :]),
            mask=mask_n[:, None],
            other=0.0,
        ).to(tl.float32)

        # Attention scores  [BLOCK_M, BLOCK_N]
        s = tl.dot(q, tl.trans(k)) * scale
        s = tl.where(mask_n[None, :], s, float('-inf'))

        # Online softmax update
        m_ij   = tl.max(s, axis=1)                # [BLOCK_M]
        m_new  = tl.maximum(m_i, m_ij)
        alpha  = tl.exp(m_i - m_new)              # rescale factor for old acc
        p      = tl.exp(s - m_new[:, None])        # [BLOCK_M, BLOCK_N]
        l_ij   = tl.sum(p, axis=1)                # [BLOCK_M]

        # V block [BLOCK_N, HEAD_DIM]
        v = tl.load(
            proj_ptr + offs_n[:, None] * PROJ_STRIDE + (v_col + offs_d[None, :]),
            mask=mask_n[:, None],
            other=0.0,
        ).to(tl.float32)

        # Accumulate
        acc = acc * alpha[:, None] + tl.dot(p, v)
        m_i = m_new
        l_i = alpha * l_i + l_ij

    # Final normalisation
    l_safe = tl.where(l_i == 0.0, 1.0, l_i)
    acc = acc / l_safe[:, None]

    # Write to out:  out[s, head_id*HEAD_DIM + d]
    out_col = head_id * HEAD_DIM
    tl.store(
        out_ptr + offs_m[:, None] * EMBED_DIM + (out_col + offs_d[None, :]),
        acc.to(out_ptr.dtype.element_ty),
        mask=offs_m[:, None] < T_B,
    )


# ---------------------------------------------------------------------------
# Host wrapper
# ---------------------------------------------------------------------------
@torch.fx.wrap
def mha_triton_forward(in_0, in_1, in_2, in_3, in_4):
    """
    in_0 : out_proj_bias    [E]
    in_1 : out_proj_weight  [E, E]
    in_2 : in_proj_bias     [3E]
    in_3 : in_proj_weight   [3E, E]
    in_4 : input            [T, B, E]
    """
    T, B, E = in_4.shape   # 150, 1, 512
    H        = 8
    HD       = E // H      # 64
    TB       = T * B       # 150

    # ---- 1. flatten input to [TB, E] ----
    x = in_4.view(TB, E)

    # ---- 2. in-projection: [TB, E] @ [3E, E]^T + [3E] -> [TB, 3E] ----
    proj = torch.empty((TB, 3 * E), dtype=in_4.dtype, device=in_4.device)
    linear_bias_kernel[
        lambda meta: (triton.cdiv(TB, meta['BLOCK_M']),
                      triton.cdiv(3 * E, meta['BLOCK_N']))
    ](
        x, in_3, in_2, proj,
        TB, 3 * E, E,
        x.stride(0),    x.stride(1),
        in_3.stride(0), in_3.stride(1),
        proj.stride(0), proj.stride(1),
    )

    # ---- 3. Flash Attention: [TB, 3E] -> [TB, E] ----
    attn_out = torch.empty((TB, E), dtype=in_4.dtype, device=in_4.device)
    scale    = 1.0 / math.sqrt(HD)

    flash_attn_kernel[
        lambda meta: (H, triton.cdiv(TB, meta['BLOCK_M']))
    ](
        proj, attn_out,
        TB,
        scale,
        EMBED_DIM=E,
        HEAD_DIM=HD,
    )

    # ---- 4. out-projection: [TB, E] @ [E, E]^T + [E] -> [TB, E] ----
    out = torch.empty((TB, E), dtype=in_4.dtype, device=in_4.device)
    linear_bias_kernel[
        lambda meta: (triton.cdiv(TB, meta['BLOCK_M']),
                      triton.cdiv(E,  meta['BLOCK_N']))
    ](
        attn_out, in_1, in_0, out,
        TB, E, E,
        attn_out.stride(0), attn_out.stride(1),
        in_1.stride(0),     in_1.stride(1),
        out.stride(0),      out.stride(1),
    )

    return out.view(T, B, E)


# ---------------------------------------------------------------------------
# Pattern / replacement API required by the framework
# ---------------------------------------------------------------------------
def pattern(in_0, in_1, in_2, in_3, in_4):
    mha = torch.nn.functional.multi_head_attention_forward(
        in_4, in_4, in_4, 512, 8, in_3, in_2, None, None, False, 0.0, in_1, in_0,
        training=False, key_padding_mask=None, need_weights=True,
        attn_mask=None, average_attn_weights=True, is_causal=False,
    )
    tmp_5 = mha[0]
    tmp_6 = torch.nn.functional.dropout(tmp_5, 0.0, False, False)
    tmp_7 = torch.nn.functional.dropout(tmp_6, 0.0, False, False)
    return tmp_7


def replacement_args(in_0, in_1, in_2, in_3, in_4):
    return (in_0, in_1, in_2, in_3, in_4)


def replacement_func():
    return mha_triton_forward