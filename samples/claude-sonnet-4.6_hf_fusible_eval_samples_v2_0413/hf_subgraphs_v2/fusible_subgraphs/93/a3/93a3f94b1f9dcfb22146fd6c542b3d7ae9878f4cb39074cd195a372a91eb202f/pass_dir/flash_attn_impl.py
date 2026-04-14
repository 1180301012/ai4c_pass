import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 64},  num_stages=2, num_warps=4),
        triton.Config({'BLOCK_M': 32,  'BLOCK_N': 64},  num_stages=2, num_warps=4),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 32},  num_stages=2, num_warps=4),
        triton.Config({'BLOCK_M': 32,  'BLOCK_N': 32},  num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 32},  num_stages=2, num_warps=4),
        triton.Config({'BLOCK_M': 32,  'BLOCK_N': 128}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64},  num_stages=2, num_warps=8),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 128}, num_stages=2, num_warps=8),
    ],
    key=['N', 'D'],
)
@triton.jit
def _flash_attn_fwd(
    Q_ptr, K_ptr, V_ptr, O_ptr,
    # Q: [B, H, N, D]
    stride_qb, stride_qh, stride_qn, stride_qd,
    # K (pre-transposed): [B, H, D, N]
    stride_kb, stride_kh, stride_kd, stride_kn,
    # V: [B, H, N, D]
    stride_vb, stride_vh, stride_vn, stride_vd,
    # O: [B, N, H*D]
    stride_ob, stride_on, stride_od,
    B, H, N, D,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,  # next power-of-2 >= D (128 for D=80)
    OUTPUT_DTYPE: tl.constexpr,  # 0=f32, 1=bf16, 2=f16
):
    prog_id = tl.program_id(0)
    num_q_blocks = tl.cdiv(N, BLOCK_M)

    b_idx       = prog_id // (H * num_q_blocks)
    h_idx       = (prog_id // num_q_blocks) % H
    q_block_idx = prog_id % num_q_blocks
    q_start     = q_block_idx * BLOCK_M

    q_range = q_start + tl.arange(0, BLOCK_M)  # [BLOCK_M]
    d_range = tl.arange(0, BLOCK_D)             # [BLOCK_D]

    # ── Load Q block [BLOCK_M, BLOCK_D] ──────────────────────────────────
    q_ptrs = (Q_ptr
              + b_idx * stride_qb
              + h_idx * stride_qh
              + q_range[:, None] * stride_qn
              + d_range[None, :] * stride_qd)
    q_mask = (q_range[:, None] < N) & (d_range[None, :] < D)
    Q = tl.load(q_ptrs, mask=q_mask, other=0.0).to(tl.float32)

    # ── Online-softmax state ──────────────────────────────────────────────
    O_acc = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)
    m     = tl.full([BLOCK_M], float('-inf'), dtype=tl.float32)
    l     = tl.zeros([BLOCK_M], dtype=tl.float32)

    # ── Loop over KV blocks ───────────────────────────────────────────────
    for kv_block in range(tl.cdiv(N, BLOCK_N)):
        kv_start = kv_block * BLOCK_N
        kv_range = kv_start + tl.arange(0, BLOCK_N)  # [BLOCK_N]

        # Load K [BLOCK_D, BLOCK_N] — stored as [B,H,D,N]
        k_ptrs = (K_ptr
                  + b_idx * stride_kb
                  + h_idx * stride_kh
                  + d_range[:, None]  * stride_kd
                  + kv_range[None, :] * stride_kn)
        k_mask = (d_range[:, None] < D) & (kv_range[None, :] < N)
        K = tl.load(k_ptrs, mask=k_mask, other=0.0).to(tl.float32)

        # S = Q @ K  →  [BLOCK_M, BLOCK_N]
        S = tl.dot(Q, K, allow_tf32=True)

        # Mask padding positions to -inf so they don't affect softmax
        S = tl.where(kv_range[None, :] < N, S, float('-inf'))

        # Online softmax update
        m_new = tl.maximum(m, tl.max(S, axis=1))          # [BLOCK_M]
        alpha  = tl.exp(m - m_new)                         # rescale old state
        p      = tl.exp(S - m_new[:, None])                # [BLOCK_M, BLOCK_N]
        l      = alpha * l + tl.sum(p, axis=1)
        m      = m_new

        # Load V [BLOCK_N, BLOCK_D] — stored as [B,H,N,D]
        v_ptrs = (V_ptr
                  + b_idx * stride_vb
                  + h_idx * stride_vh
                  + kv_range[:, None] * stride_vn
                  + d_range[None, :]  * stride_vd)
        v_mask = (kv_range[:, None] < N) & (d_range[None, :] < D)
        V = tl.load(v_ptrs, mask=v_mask, other=0.0).to(tl.float32)

        # Accumulate: O = diag(alpha) * O + p @ V
        O_acc = O_acc * alpha[:, None] + tl.dot(p, V, allow_tf32=True)

    # ── Normalize ─────────────────────────────────────────────────────────
    O_final = O_acc / l[:, None]

    # ── Store to output [B, N, H*D] ───────────────────────────────────────
    # Each head h writes to columns [h*D .. (h+1)*D - 1]
    out_col  = h_idx * D + d_range                          # [BLOCK_D]
    out_ptrs = (O_ptr
                + b_idx * stride_ob
                + q_range[:, None] * stride_on
                + out_col[None, :] * stride_od)
    # Only write the D valid columns (d_range < D), not the padding
    out_mask = (q_range[:, None] < N) & (d_range[None, :] < D)

    if OUTPUT_DTYPE == 1:   # bfloat16
        tl.store(out_ptrs, O_final.to(tl.bfloat16), mask=out_mask)
    elif OUTPUT_DTYPE == 2: # float16
        tl.store(out_ptrs, O_final.to(tl.float16),  mask=out_mask)
    else:                   # float32
        tl.store(out_ptrs, O_final,                  mask=out_mask)


# BLOCK_D: next power-of-2 >= head-dim D=80
_BLOCK_D = 128


@torch.fx.wrap
def flash_attn_dispatch(in_0, in_1, in_2, route):
    """Single dispatch wrapper for all dtype variants.
    route: "bf16" | "f16" | "f32"
    All pass files import and return THIS same function object so that
    output_pass_replacement_func_limit treats them identically.
    """
    B = in_0.shape[0]
    H = in_0.shape[1]
    N = in_0.shape[2]
    D = in_0.shape[3]
    grid = lambda meta: (B * H * triton.cdiv(N, meta['BLOCK_M']),)

    if route == "bf16":
        O = torch.empty((B, N, H * D), dtype=torch.bfloat16, device=in_0.device)
        _flash_attn_fwd[grid](
            in_0, in_1, in_2, O,
            in_0.stride(0), in_0.stride(1), in_0.stride(2), in_0.stride(3),
            in_1.stride(0), in_1.stride(1), in_1.stride(2), in_1.stride(3),
            in_2.stride(0), in_2.stride(1), in_2.stride(2), in_2.stride(3),
            O.stride(0), O.stride(1), O.stride(2),
            B, H, N, D,
            BLOCK_D=_BLOCK_D, OUTPUT_DTYPE=1,
        )
    elif route == "f16":
        O = torch.empty((B, N, H * D), dtype=torch.float16, device=in_0.device)
        _flash_attn_fwd[grid](
            in_0, in_1, in_2, O,
            in_0.stride(0), in_0.stride(1), in_0.stride(2), in_0.stride(3),
            in_1.stride(0), in_1.stride(1), in_1.stride(2), in_1.stride(3),
            in_2.stride(0), in_2.stride(1), in_2.stride(2), in_2.stride(3),
            O.stride(0), O.stride(1), O.stride(2),
            B, H, N, D,
            BLOCK_D=_BLOCK_D, OUTPUT_DTYPE=2,
        )
    else:  # "f32"
        O = torch.empty((B, N, H * D), dtype=torch.float32, device=in_0.device)
        _flash_attn_fwd[grid](
            in_0, in_1, in_2, O,
            in_0.stride(0), in_0.stride(1), in_0.stride(2), in_0.stride(3),
            in_1.stride(0), in_1.stride(1), in_1.stride(2), in_1.stride(3),
            in_2.stride(0), in_2.stride(1), in_2.stride(2), in_2.stride(3),
            O.stride(0), O.stride(1), O.stride(2),
            B, H, N, D,
            BLOCK_D=_BLOCK_D, OUTPUT_DTYPE=0,
        )
    return O