import torch
import triton
import triton.language as tl
import math


# ---------------------------------------------------------------------------
# GEMM kernel: C = A @ B_T + bias
#   A  : [M, K]  (row-major)
#   B_T: [N, K]  (weight matrix, stored row-major; we compute A @ B_T.T = A @ W)
#   C  : [M, N]
# ---------------------------------------------------------------------------
@triton.jit
def gemm_nt_bias_kernel(
    A_ptr, B_ptr, bias_ptr, C_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bn, stride_bk,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    HAS_BIAS: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k_start in range(0, K, BLOCK_K):
        offs_k = k_start + tl.arange(0, BLOCK_K)

        A_ptrs = A_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
        B_ptrs = B_ptr + offs_n[:, None] * stride_bn + offs_k[None, :] * stride_bk

        a_mask = (offs_m[:, None] < M) & (offs_k[None, :] < K)
        b_mask = (offs_n[:, None] < N) & (offs_k[None, :] < K)

        a = tl.load(A_ptrs, mask=a_mask, other=0.0).to(tl.float32)
        b = tl.load(B_ptrs, mask=b_mask, other=0.0).to(tl.float32)

        # a: [BLOCK_M, BLOCK_K] @ tl.trans(b): [BLOCK_K, BLOCK_N] -> [BLOCK_M, BLOCK_N]
        acc = tl.dot(a, tl.trans(b), acc=acc, allow_tf32=True)

    if HAS_BIAS:
        bias = tl.load(bias_ptr + offs_n, mask=offs_n < N, other=0.0).to(tl.float32)
        acc = acc + bias[None, :]

    C_ptrs = C_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(C_ptrs, acc.to(C_ptr.dtype.element_ty), mask=c_mask)


# ---------------------------------------------------------------------------
# Flash-Attention-2 kernel (forward, no causal mask, no dropout)
#   Q, K, V : [H, T, D]  (contiguous)
#   O       : [H, T, D]
#   Grid    : (H, ceil(T / BLOCK_T))
# ---------------------------------------------------------------------------
@triton.jit
def flash_attn_fwd_kernel(
    Q_ptr, K_ptr, V_ptr, O_ptr,
    T, D,
    scale,
    stride_qh, stride_qt, stride_qd,
    stride_kh, stride_kt, stride_kd,
    stride_vh, stride_vt, stride_vd,
    stride_oh, stride_ot, stride_od,
    BLOCK_T: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    h_idx   = tl.program_id(0)
    q_block = tl.program_id(1)

    q_start = q_block * BLOCK_T
    offs_q  = q_start + tl.arange(0, BLOCK_T)   # query positions
    offs_d  = tl.arange(0, BLOCK_D)              # head-dim positions

    # Load Q block: [BLOCK_T, BLOCK_D]
    Q_ptrs = Q_ptr + h_idx * stride_qh + offs_q[:, None] * stride_qt + offs_d[None, :] * stride_qd
    q_mask = (offs_q[:, None] < T) & (offs_d[None, :] < D)
    q = tl.load(Q_ptrs, mask=q_mask, other=0.0).to(tl.float32)

    # Running stats for online softmax
    m_i = tl.full((BLOCK_T,), float('-inf'), dtype=tl.float32)
    l_i = tl.zeros((BLOCK_T,),              dtype=tl.float32)
    acc = tl.zeros((BLOCK_T, BLOCK_D),      dtype=tl.float32)

    # Iterate over all K/V tiles
    for kv_start in range(0, T, BLOCK_T):
        offs_kv  = kv_start + tl.arange(0, BLOCK_T)
        kv_mask  = (offs_kv[:, None] < T) & (offs_d[None, :] < D)

        # Load K tile: [BLOCK_T, BLOCK_D]
        K_ptrs = K_ptr + h_idx * stride_kh + offs_kv[:, None] * stride_kt + offs_d[None, :] * stride_kd
        k = tl.load(K_ptrs, mask=kv_mask, other=0.0).to(tl.float32)

        # Attention scores: Q @ K^T -> [BLOCK_T, BLOCK_T]
        s = tl.dot(q, tl.trans(k), allow_tf32=True) * scale
        # Mask out-of-range kv positions
        s = tl.where(offs_kv[None, :] < T, s, float('-inf'))

        # --- online softmax update ---
        m_new = tl.maximum(m_i, tl.max(s, axis=1))
        p     = tl.exp(s - m_new[:, None])           # [BLOCK_T, BLOCK_T]
        alpha = tl.exp(m_i - m_new)                  # [BLOCK_T]
        l_i   = l_i * alpha + tl.sum(p, axis=1)

        # Load V tile: [BLOCK_T, BLOCK_D]
        V_ptrs = V_ptr + h_idx * stride_vh + offs_kv[:, None] * stride_vt + offs_d[None, :] * stride_vd
        v = tl.load(V_ptrs, mask=kv_mask, other=0.0).to(tl.float32)

        # p @ V -> [BLOCK_T, BLOCK_D]
        pv  = tl.dot(p, v, allow_tf32=True)
        acc = acc * alpha[:, None] + pv

        m_i = m_new

    # Normalise
    acc = acc / l_i[:, None]

    # Store output
    O_ptrs = O_ptr + h_idx * stride_oh + offs_q[:, None] * stride_ot + offs_d[None, :] * stride_od
    tl.store(O_ptrs, acc.to(O_ptr.dtype.element_ty), mask=q_mask)


# ---------------------------------------------------------------------------
# Orchestration wrapper
# ---------------------------------------------------------------------------
@torch.fx.wrap
def fused_mha_triton(x, in_proj_weight, in_proj_bias, out_proj_weight, out_proj_bias):
    """
    Fused self-attention (q=k=v=x) forward pass.
    x              : [T, B, E]   typically [150, 1, 512]
    in_proj_weight : [3E, E]     = [1536, 512]
    in_proj_bias   : [3E]        = [1536]
    out_proj_weight: [E, E]      = [512, 512]
    out_proj_bias  : [E]         = [512]
    Returns        : [T, B, E]
    """
    T, B, E = x.shape
    H  = 8
    D  = E // H          # 64
    TB = T * B           # 150 (for batch=1)

    dtype  = x.dtype
    device = x.device

    BLOCK_M_G  = 32
    BLOCK_N_G  = 64
    BLOCK_K_G  = 64

    # ------------------------------------------------------------------ #
    # Step 1 : QKV in-projection  [TB, E] @ [3E, E].T -> [TB, 3E]
    # ------------------------------------------------------------------ #
    x_2d = x.reshape(TB, E)                                  # [150, 512]
    qkv  = torch.empty((TB, 3 * E), dtype=dtype, device=device)

    grid_p1 = (triton.cdiv(TB, BLOCK_M_G), triton.cdiv(3 * E, BLOCK_N_G))
    gemm_nt_bias_kernel[grid_p1](
        x_2d, in_proj_weight, in_proj_bias, qkv,
        TB, 3 * E, E,
        x_2d.stride(0), x_2d.stride(1),
        in_proj_weight.stride(0), in_proj_weight.stride(1),
        qkv.stride(0), qkv.stride(1),
        BLOCK_M=BLOCK_M_G, BLOCK_N=BLOCK_N_G, BLOCK_K=BLOCK_K_G,
        HAS_BIAS=True,
    )

    # ------------------------------------------------------------------ #
    # Step 2 : Split + reshape to [H, TB, D] contiguous
    # ------------------------------------------------------------------ #
    # qkv[:, :E] has strides (3E, 1) -> non-contiguous; reshape makes copy
    q = qkv[:, :E     ].reshape(TB, H, D).permute(1, 0, 2).contiguous()   # [8, 150, 64]
    k = qkv[:, E:2*E  ].reshape(TB, H, D).permute(1, 0, 2).contiguous()
    v = qkv[:, 2*E:   ].reshape(TB, H, D).permute(1, 0, 2).contiguous()

    # ------------------------------------------------------------------ #
    # Step 3 : Flash Attention  -> o [H, TB, D]
    # ------------------------------------------------------------------ #
    o     = torch.empty((H, TB, D), dtype=dtype, device=device)
    scale = 1.0 / math.sqrt(D)

    BLOCK_T_A = 32   # must be power-of-2 >= 16
    BLOCK_D_A = 64   # must equal D

    grid_attn = (H, triton.cdiv(TB, BLOCK_T_A))
    flash_attn_fwd_kernel[grid_attn](
        q, k, v, o,
        TB, D, scale,
        q.stride(0), q.stride(1), q.stride(2),
        k.stride(0), k.stride(1), k.stride(2),
        v.stride(0), v.stride(1), v.stride(2),
        o.stride(0), o.stride(1), o.stride(2),
        BLOCK_T=BLOCK_T_A, BLOCK_D=BLOCK_D_A,
    )

    # ------------------------------------------------------------------ #
    # Step 4 : Merge heads  [H, TB, D] -> [TB, E]
    # ------------------------------------------------------------------ #
    context = o.permute(1, 0, 2).contiguous().reshape(TB, E)   # [150, 512]

    # ------------------------------------------------------------------ #
    # Step 5 : Out-projection  [TB, E] @ [E, E].T -> [TB, E]
    # ------------------------------------------------------------------ #
    output = torch.empty((TB, E), dtype=dtype, device=device)

    grid_p2 = (triton.cdiv(TB, BLOCK_M_G), triton.cdiv(E, BLOCK_N_G))
    gemm_nt_bias_kernel[grid_p2](
        context, out_proj_weight, out_proj_bias, output,
        TB, E, E,
        context.stride(0), context.stride(1),
        out_proj_weight.stride(0), out_proj_weight.stride(1),
        output.stride(0), output.stride(1),
        BLOCK_M=BLOCK_M_G, BLOCK_N=BLOCK_N_G, BLOCK_K=BLOCK_K_G,
        HAS_BIAS=True,
    )

    return output.reshape(T, B, E)


# ---------------------------------------------------------------------------
# Lightweight replacement: fuse getitem(mha_tuple, 0) + 2 identity dropouts
# into a single Python call that just returns the attention-output tensor.
# dropout(p=0.0, training=False) is a C++-level no-op; every avoided kernel
# dispatch saves wall-clock time.
# ---------------------------------------------------------------------------
@triton.jit
def identity_copy_kernel(in_ptr, out_ptr, N, BLOCK: tl.constexpr):
    pid  = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    tl.store(out_ptr + offs, tl.load(in_ptr + offs, mask=mask), mask=mask)


@torch.fx.wrap
def fused_getitem_dropout(result_tuple):
    """
    Fuse: getitem(mha_result, 0)
          dropout(attn_out, 0.0, False, False)   <- identity (p=0 → no-op)
          dropout(attn_out, 0.0, False, False)   <- identity (p=0 → no-op)
    All three are pure identity: just extract and return the tensor directly.
    Eliminates two C++/CUDA dispatch overheads with zero memory allocation.
    """
    # Plain Python tuple indexing — no torch.* call, no allocation, no copy.
    return result_tuple[0]


# ---------------------------------------------------------------------------
# Pattern / replacement API
# ---------------------------------------------------------------------------

def pattern(result_tuple):
    """
    Match the 3-node suffix:
        getitem(mha_result, 0)
        dropout(attn_out, 0.0, False, False)
        dropout(attn_out, 0.0, False, False)
    'result_tuple' is the tuple returned by multi_head_attention_forward.
    """
    output = result_tuple[0]
    d1 = torch.nn.functional.dropout(output, 0.0, False, False)
    d2 = torch.nn.functional.dropout(d1, 0.0, False, False)
    return d2


def replacement_args(result_tuple):
    return (result_tuple,)


def replacement_func():
    return fused_getitem_dropout