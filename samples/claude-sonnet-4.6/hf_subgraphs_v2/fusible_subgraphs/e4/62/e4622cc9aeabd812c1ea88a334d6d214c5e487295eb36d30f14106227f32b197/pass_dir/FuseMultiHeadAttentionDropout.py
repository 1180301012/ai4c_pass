import torch
import triton
import triton.language as tl
import math


# ============================================================
# Kernel 1: GEMM C = A @ W.T + Bias
#   A: [M, K],  W: [N, K],  Bias: [N],  C: [M, N]
# ============================================================
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64,  'BLOCK_K': 32}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 64,  'BLOCK_K': 64}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 64,  'BLOCK_K': 32}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64,  'BLOCK_K': 64}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 128, 'BLOCK_K': 64}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_warps=8, num_stages=4),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def gemm_nt_bias_kernel(
    A, W, Bias, C,
    M, N, K,
    stride_am, stride_ak,
    stride_wn, stride_wk,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
    for k_start in range(0, K, BLOCK_K):
        offs_k = k_start + tl.arange(0, BLOCK_K)
        a = tl.load(
            A + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak,
            mask=(offs_m[:, None] < M) & (offs_k[None, :] < K), other=0.0
        )
        w = tl.load(
            W + offs_n[:, None] * stride_wn + offs_k[None, :] * stride_wk,
            mask=(offs_n[:, None] < N) & (offs_k[None, :] < K), other=0.0
        )
        acc += tl.dot(a.to(tl.float32), tl.trans(w).to(tl.float32))

    bias = tl.load(Bias + offs_n, mask=offs_n < N, other=0.0)
    acc += bias[None, :].to(tl.float32)

    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(
        C + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn,
        acc.to(C.dtype.element_ty), mask=mask,
    )


# ============================================================
# Kernel 2: Flash Attention over packed QKV
#
#   QKV : [M, 3*N_PROJ]  (Q cols=[0:N_PROJ], K=[N_PROJ:2N], V=[2N:3N])
#   Out : [M, N_PROJ]    (laid out as [M, H*HEAD_DIM])
#
#   Grid: (cdiv(M, BLOCK_M), H)
# ============================================================
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64}, num_warps=8, num_stages=2),
    ],
    key=['M'],
)
@triton.jit
def flash_attn_packed_qkv_kernel(
    QKV,
    Out,
    M,            # sequence length
    scale,
    stride_qkv_m, # stride along seq dim in QKV  (= 3*N_PROJ)
    stride_out_m, # stride along seq dim in Out   (= N_PROJ)
    N_PROJ: tl.constexpr,    # embed_dim = 512
    HEAD_DIM: tl.constexpr,  # head_dim  = 64
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """
    Each program handles BLOCK_M query tokens for one head (pid_h = program_id(1)).
    Q for head h lives at QKV[:, h*HEAD_DIM : (h+1)*HEAD_DIM]
    K for head h lives at QKV[:, N_PROJ + h*HEAD_DIM : N_PROJ + (h+1)*HEAD_DIM]
    V for head h lives at QKV[:, 2*N_PROJ + h*HEAD_DIM : 2*N_PROJ + (h+1)*HEAD_DIM]
    Output head h lives at Out[:, h*HEAD_DIM : (h+1)*HEAD_DIM]
    """
    pid_m = tl.program_id(0)
    pid_h = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, HEAD_DIM)

    # Base offsets inside QKV for Q, K, V of this head
    q_base = pid_h * HEAD_DIM
    k_base = N_PROJ + pid_h * HEAD_DIM
    v_base = 2 * N_PROJ + pid_h * HEAD_DIM

    q_mask = offs_m[:, None] < M
    q = tl.load(
        QKV + q_base + offs_m[:, None] * stride_qkv_m + offs_d[None, :],
        mask=q_mask, other=0.0
    ).to(tl.float32)

    # Running statistics for online softmax
    m_i = tl.full([BLOCK_M], float('-inf'), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)

    # Iterate over key / value blocks
    for n_start in range(0, M, BLOCK_N):
        offs_n = n_start + tl.arange(0, BLOCK_N)
        kv_mask = offs_n[:, None] < M

        k = tl.load(
            QKV + k_base + offs_n[:, None] * stride_qkv_m + offs_d[None, :],
            mask=kv_mask, other=0.0
        ).to(tl.float32)

        v = tl.load(
            QKV + v_base + offs_n[:, None] * stride_qkv_m + offs_d[None, :],
            mask=kv_mask, other=0.0
        ).to(tl.float32)

        # Attention scores  [BLOCK_M, BLOCK_N]
        qk = tl.dot(q, tl.trans(k)) * scale
        # Mask out-of-range positions
        qk = tl.where(offs_n[None, :] < M, qk, float('-inf'))

        # Online softmax update
        m_new = tl.maximum(m_i, tl.max(qk, axis=1))
        p = tl.exp(qk - m_new[:, None])
        l_new = l_i * tl.exp(m_i - m_new) + tl.sum(p, axis=1)

        # Accumulate weighted values  [BLOCK_M, HEAD_DIM]
        acc = acc * tl.exp(m_i - m_new)[:, None] + tl.dot(p, v)

        m_i = m_new
        l_i = l_new

    # Normalize
    acc = acc / l_i[:, None]

    # Store to Out[m, h*HEAD_DIM + d]
    out_base = pid_h * HEAD_DIM
    tl.store(
        Out + out_base + offs_m[:, None] * stride_out_m + offs_d[None, :],
        acc.to(Out.dtype.element_ty),
        mask=q_mask,
    )


# ============================================================
# Pattern: getitem[0] + two consecutive no-op dropout calls
# Matches: MHA_tuple[0] → dropout(p=0) → dropout(p=0)
# ============================================================
def pattern(x):
    tmp = x[0]                                                    # getitem
    tmp_6 = torch.nn.functional.dropout(tmp,   0.0, False, False)
    tmp_7 = torch.nn.functional.dropout(tmp_6, 0.0, False, False)
    return tmp_7


def replacement_args(x):
    # x = the MHA output tuple; we just need to pass it through
    return (x,)


# ============================================================
# Optimised replacement: identity extraction
# dropout(p=0) is a no-op → return x[0] directly
# ============================================================
@triton.jit
def _noop_kernel(x_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid  = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    _    = tl.load(x_ptr + offs, mask=offs < n_elements, other=0.0)


@torch.fx.wrap
def optimized_mha_forward(x):
    # x is the MHA tuple (attn_output, attn_weights)
    # Both dropouts are p=0 no-ops; extract first element directly
    return x[0]


def replacement_func():
    return optimized_mha_forward