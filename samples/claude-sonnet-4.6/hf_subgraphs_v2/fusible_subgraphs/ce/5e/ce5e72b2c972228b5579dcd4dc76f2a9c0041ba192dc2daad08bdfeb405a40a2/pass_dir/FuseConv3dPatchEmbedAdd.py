import torch
import triton
import triton.language as tl
from torch import device as torch_device


@triton.autotune(
    configs=[
        # Wide N=128 tiles — good for N=768
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 128, 'BLOCK_K': 32,  'GROUP_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 128, 'BLOCK_K': 64,  'GROUP_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32,  'GROUP_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64,  'GROUP_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64,  'GROUP_M': 8}, num_stages=5, num_warps=4),
        # Balanced M=N=64 — more blocks → better SM occupancy
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 64,  'BLOCK_K': 32,  'GROUP_M': 8}, num_stages=5, num_warps=4),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 64,  'BLOCK_K': 64,  'GROUP_M': 8}, num_stages=5, num_warps=4),
        # Tall-M tiles
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64,  'BLOCK_K': 32,  'GROUP_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64,  'BLOCK_K': 64,  'GROUP_M': 8}, num_stages=4, num_warps=4),
        # Higher warp counts for tensor-core throughput on Ampere
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 128, 'BLOCK_K': 32,  'GROUP_M': 8}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32,  'GROUP_M': 8}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64,  'BLOCK_K': 32,  'GROUP_M': 8}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64,  'GROUP_M': 8}, num_stages=3, num_warps=8),
        # Large BLOCK_K — fewer loop trips
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 64,  'BLOCK_K': 128, 'GROUP_M': 8}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64,  'BLOCK_K': 128, 'GROUP_M': 8}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 128, 'BLOCK_K': 128, 'GROUP_M': 8}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 128, 'BLOCK_K': 128, 'GROUP_M': 8}, num_stages=4, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 128, 'GROUP_M': 8}, num_stages=3, num_warps=8),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def patch_embed_gemm_bias_add_kernel(
    input_ptr,    # [B, C, T, H, W] — contiguous, B=1
    weight_ptr,   # [N, K]  (contiguous from [N, C, KT, KH, KW])
    bias_ptr,     # [N]
    pos_emb_ptr,  # [*, M, N] on GPU, same dtype
    output_ptr,   # [M, N]  (output)
    # GEMM dimensions
    M, N, K,
    # Spatial output dimensions
    H_OUT, W_OUT,
    # Kernel spatial parameters (compile-time constants)
    KT: tl.constexpr,
    KH: tl.constexpr,
    KW: tl.constexpr,
    # Input strides (contiguous NCTHW)
    stride_in_c, stride_in_t, stride_in_h, stride_in_w,
    # Tile sizes (auto-tuned)
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    # Derived compile-time constants
    KTHW: tl.constexpr = KT * KH * KW     # = 512
    KHW:  tl.constexpr = KH * KW          # = 256

    # ── Block-level 2D coordinate with grouped swizzle for L2 re-use ──────
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_M * num_pid_n
    group_id        = pid // num_pid_in_group
    first_pid_m     = group_id * GROUP_M
    group_size_m    = min(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + (pid % num_pid_in_group) % group_size_m
    pid_n = (pid % num_pid_in_group) // group_size_m

    m_start = pid_m * BLOCK_M
    n_start = pid_n * BLOCK_N

    m_offs = m_start + tl.arange(0, BLOCK_M)   # [BLOCK_M]
    n_offs = n_start + tl.arange(0, BLOCK_N)   # [BLOCK_N]

    # Decode patch index m → spatial position (t, h, w)
    HW_OUT = H_OUT * W_OUT
    m_t  = m_offs // HW_OUT
    m_hw = m_offs % HW_OUT
    m_h  = m_hw // W_OUT
    m_w  = m_hw % W_OUT

    # Masks
    m_mask = m_offs < M   # [BLOCK_M]
    n_mask = n_offs < N   # [BLOCK_N]

    # ── Hoist loop-invariant m-dependent base address ──────────────────────
    m_base = (m_t * (KT * stride_in_t) +
              m_h * (KH * stride_in_h) +
              m_w * KW)                    # [BLOCK_M]; stride_in_w == 1

    # ── Main GEMM accumulation loop — float32 accumulator ──
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k_blk in range(0, tl.cdiv(K, BLOCK_K)):
        k_offs = k_blk * BLOCK_K + tl.arange(0, BLOCK_K)   # [BLOCK_K]
        k_mask = k_offs < K

        # Decode kernel index k → (c, dt, dh, dw)
        k_c    = k_offs // KTHW
        k_rem  = k_offs % KTHW
        k_dt   = k_rem  // KHW
        k_rem2 = k_rem  % KHW
        k_dh   = k_rem2 // KW
        k_dw   = k_rem2 % KW

        # k-dependent input offset
        k_base = (k_c   * stride_in_c +
                  k_dt  * stride_in_t +
                  k_dh  * stride_in_h +
                  k_dw)                    # [BLOCK_K]

        # Input gather: [BLOCK_M, BLOCK_K]
        in_idx = m_base[:, None] + k_base[None, :]
        a = tl.load(input_ptr + in_idx,
                    mask=m_mask[:, None] & k_mask[None, :],
                    other=0.0)

        # Weight tile: [BLOCK_N, BLOCK_K] — contiguous access
        w_idx = n_offs[:, None] * K + k_offs[None, :]
        b = tl.load(weight_ptr + w_idx,
                    mask=n_mask[:, None] & k_mask[None, :],
                    other=0.0)

        # Tensor-core GEMM accumulation (f16/bf16) or TF32 (f32)
        acc = tl.dot(a, tl.trans(b), acc, allow_tf32=True)

    # ── Fused bias add ──
    bias = tl.load(bias_ptr + n_offs, mask=n_mask, other=0.0).to(tl.float32)
    acc += bias[None, :]

    # ── Fused positional-embedding add ──
    out_mask = m_mask[:, None] & n_mask[None, :]
    pos_idx  = m_offs[:, None] * N + n_offs[None, :]
    pos = tl.load(pos_emb_ptr + pos_idx, mask=out_mask, other=0.0).to(tl.float32)
    acc += pos

    # ── Store ──
    tl.store(output_ptr + pos_idx, acc, mask=out_mask)


# ─────────────────────────────────────────────────────────────────────────────
# Pattern to match
# ─────────────────────────────────────────────────────────────────────────────
def pattern(in_0, in_1, in_2, in_3):
    conv3d = torch.conv3d(in_3, in_1, in_0, (2, 16, 16), (0, 0, 0), (1, 1, 1), 1)
    tmp_4  = conv3d.flatten(2)
    tmp_5  = tmp_4.transpose(1, 2)
    tmp_6  = in_2.detach()
    tmp_7  = tmp_6.type_as(tmp_5)
    tmp_8  = tmp_7.to(device=torch_device(type='cuda', index=0), copy=True)
    tmp_9  = tmp_5 + tmp_8
    return tmp_9


def replacement_args(in_0, in_1, in_2, in_3):
    return (in_0, in_1, in_2, in_3)


# ─────────────────────────────────────────────────────────────────────────────
# Kernel wrapper
# ─────────────────────────────────────────────────────────────────────────────
@torch.fx.wrap
def patch_embed_fused(in_0, in_1, in_2, in_3):
    """
    Fused replacement for:
        conv3d(in_3, in_1, in_0, stride=(2,16,16)) → flatten(2) → transpose(1,2)
        + in_2 [CPU position embedding]
    Returns: [1, M, N]  (same as original output)
    """
    # ── Shapes (all metadata accesses — no dispatch) ────────────────────────
    B, C, T, H, W = in_3.shape
    N_out = in_1.shape[0]          # 768
    KT, KH, KW = 2, 16, 16        # fixed kernel / stride sizes

    T_out = (T - KT) // KT + 1
    H_out = (H - KH) // KH + 1
    W_out = (W - KW) // KW + 1
    M = T_out * H_out * W_out      # 1568
    N = N_out                      # 768
    K = C * KT * KH * KW          # 1536

    # ── Prepare position embeddings ─────────────────────────────────────────
    # torch.as_tensor is whitelisted; it copies in_2 to GPU with matching dtype.
    # We pass the resulting [1, M, N] tensor directly — the kernel indexes as
    # pos_idx = m * N + n which correctly addresses [0, m, n] in a [1,M,N] layout.
    pos_emb = torch.as_tensor(in_2, dtype=in_3.dtype, device=in_3.device)

    # ── Input is passed directly — weight in_1 is [N, C, KT, KH, KW]
    # contiguous, so flat index n*K + k maps correctly to [n, c, dt, dh, dw].
    # Contiguous NCTHW strides (B=1, no batch-dimension offset needed)
    stride_in_c = T * H * W
    stride_in_t =     H * W
    stride_in_h =         W
    stride_in_w =             1

    # ── Output allocation ───────────────────────────────────────────────────
    out_flat = torch.empty((M, N), dtype=in_3.dtype, device=in_3.device)

    # ── Launch ──────────────────────────────────────────────────────────────
    grid = lambda META: (
        triton.cdiv(M, META['BLOCK_M']) * triton.cdiv(N, META['BLOCK_N']),
    )

    patch_embed_gemm_bias_add_kernel[grid](
        in_3,
        in_1,        # [N, C, KT, KH, KW] — contiguous, same memory as [N, K]
        in_0,
        pos_emb,     # [1, M, N] on GPU — kernel addresses as m*N+n (offset 0)
        out_flat,
        M, N, K,
        H_out, W_out,
        KT, KH, KW,
        stride_in_c, stride_in_t, stride_in_h, stride_in_w,
    )

    # out_flat is a regular torch.empty tensor — .reshape() is fine here
    return out_flat.reshape(1, M, N)


def replacement_func():
    return patch_embed_fused