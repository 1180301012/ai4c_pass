"""
Optimization pass: conv2d (1x1) + dropout(p=0.0) + residual_add → pure Triton

Key design choices for M=128, K=256, N=1024 on A30 (56 SMs):
  • BLOCK_K = K = 256  → K loop runs EXACTLY ONCE (no loop overhead)
  • Small BLOCK_M (16-32) → many programs → high SM occupancy (64-256 programs)
  • num_stages=1         → no pipeline for single-iteration kernel (saves shared mem)
  • TF32 via allow_tf32  → tensor core acceleration on Ampere

All three operations (GEMM + bias + residual) are fused in one kernel.
"""

import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Pattern to match
# ---------------------------------------------------------------------------
def pattern(in_0, in_1, in_2, in_3):
    conv = torch.conv2d(in_3, in_1, in_0, (1, 1), (0, 0), (1, 1), 1)
    drop = torch.nn.functional.dropout(conv, 0.0, False, False)
    out  = drop + in_2
    return out


def replacement_args(in_0, in_1, in_2, in_3):
    return (in_0, in_1, in_2, in_3)


# ---------------------------------------------------------------------------
# Triton kernel  – 2-D program grid, fused GEMM + bias + residual
#
# Shapes for this problem:
#   A (weight)   : [M=128, K=256]
#   B (input)    : [K=256, N=1024]
#   bias         : [M=128]
#   R (residual) : [M=128, N=1024]
#   output       : [M=128, N=1024]
# ---------------------------------------------------------------------------
@triton.autotune(
    configs=[
        # ── BLOCK_K=256: single K-iteration, no loop overhead ────────────────
        # 128 programs  (8×16)  – best for float16/bf16 tensor cores
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 64,  'BLOCK_K': 256}, num_warps=4, num_stages=1),
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 64,  'BLOCK_K': 256}, num_warps=8, num_stages=1),
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 128, 'BLOCK_K': 256}, num_warps=4, num_stages=1),
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 128, 'BLOCK_K': 256}, num_warps=8, num_stages=1),
        # 64 programs   (4×16)
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64,  'BLOCK_K': 256}, num_warps=4, num_stages=1),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64,  'BLOCK_K': 256}, num_warps=8, num_stages=1),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 128, 'BLOCK_K': 256}, num_warps=4, num_stages=1),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 128, 'BLOCK_K': 256}, num_warps=8, num_stages=1),
        # 16 programs   (2×8)
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 256}, num_warps=8, num_stages=1),
        # ── BLOCK_K=128: 2-iteration K loop, pipelined ───────────────────────
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 64,  'BLOCK_K': 128}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 128, 'BLOCK_K': 128}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64,  'BLOCK_K': 128}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 128, 'BLOCK_K': 128}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 128}, num_warps=8, num_stages=2),
        # ── BLOCK_K=64: 4-iteration K loop, well-pipelined ───────────────────
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 64,  'BLOCK_K': 64},  num_warps=4, num_stages=4),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64,  'BLOCK_K': 64},  num_warps=4, num_stages=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64,  'BLOCK_K': 64},  num_warps=4, num_stages=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 64},  num_warps=8, num_stages=4),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def fused_gemm_bias_residual_kernel(
    A_ptr,     # weight   [M, K]
    B_ptr,     # input    [K, N]
    bias_ptr,  # bias     [M]
    R_ptr,     # residual [M, N]
    out_ptr,   # output   [M, N]
    M, N, K,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    # Pointers to first (and usually only) K-tile
    A_ptrs = A_ptr + offs_m[:, None] * K + offs_k[None, :]   # [BLOCK_M, BLOCK_K]
    B_ptrs = B_ptr + offs_k[:, None] * N + offs_n[None, :]   # [BLOCK_K, BLOCK_N]

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # K loop – runs ONCE when BLOCK_K == K (no loop overhead at all)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        k_rem  = K - k * BLOCK_K
        a_mask = (offs_m[:, None] < M) & (offs_k[None, :] < k_rem)
        b_mask = (offs_k[:, None] < k_rem) & (offs_n[None, :] < N)
        a = tl.load(A_ptrs, mask=a_mask, other=0.0)
        b = tl.load(B_ptrs, mask=b_mask, other=0.0)
        # allow_tf32 enables TF32 tensor cores for fp32; fp16/bf16 use native TC
        acc += tl.dot(a, b, allow_tf32=True, out_dtype=tl.float32)
        A_ptrs += BLOCK_K
        B_ptrs += BLOCK_K * N

    # Fused bias broadcast + residual add
    bias_vals = tl.load(bias_ptr + offs_m, mask=offs_m < M, other=0.0)
    acc += bias_vals[:, None].to(tl.float32)

    out_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    r_vals   = tl.load(R_ptr + offs_m[:, None] * N + offs_n[None, :],
                        mask=out_mask, other=0.0)
    acc += r_vals.to(tl.float32)

    tl.store(out_ptr + offs_m[:, None] * N + offs_n[None, :],
             acc.to(out_ptr.dtype.element_ty), mask=out_mask)


# ---------------------------------------------------------------------------
# Specialised mask-free kernel for the EXACT problem shapes
#   M=128, K=256, N=1024 (all dims are exact multiples of any block size we use)
#
# Key advantages over the generic kernel:
#   • No boundary-check masks  → cleaner loads/stores
#   • K = BLOCK_K = 256        → no K-loop, single tl.dot instruction
#   • M, K as constexpr        → compiler can fully optimise pointer arithmetic
# ---------------------------------------------------------------------------
@triton.autotune(
    configs=[
        # ── 256 programs (8×32): max occupancy, works for all dtypes ─────────
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 32},  num_warps=2, num_stages=1),
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 32},  num_warps=4, num_stages=1),
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 32},  num_warps=8, num_stages=1),
        # ── 128 programs (8×16) ───────────────────────────────────────────────
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 64},  num_warps=4, num_stages=1),
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 64},  num_warps=8, num_stages=1),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 32},  num_warps=4, num_stages=1),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 32},  num_warps=8, num_stages=1),
        # ── 64 programs ───────────────────────────────────────────────────────
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 128}, num_warps=4, num_stages=1),
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 128}, num_warps=8, num_stages=1),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64},  num_warps=4, num_stages=1),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64},  num_warps=8, num_stages=1),
        # ── 32 programs ───────────────────────────────────────────────────────
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 128}, num_warps=4, num_stages=1),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 128}, num_warps=8, num_stages=1),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64},  num_warps=4, num_stages=1),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64},  num_warps=8, num_stages=1),
        # ── 16 programs ───────────────────────────────────────────────────────
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128}, num_warps=8, num_stages=1),
        # ── 8 programs ────────────────────────────────────────────────────────
        triton.Config({'BLOCK_M': 128,'BLOCK_N': 128}, num_warps=8, num_stages=1),
    ],
    key=['N'],   # N=HW=1024 is the only runtime key; M,K are constexpr
)
@triton.jit
def fused_gemm_nomask_kernel(
    A_ptr, B_ptr, bias_ptr, R_ptr, out_ptr,
    N,                              # runtime: N=HW (needed for strides)
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    M: tl.constexpr,               # always 128
    K: tl.constexpr,               # always 256  (= BLOCK_K)
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, K)      # K = BLOCK_K = 256: single load, no loop

    # Load entire A-tile [BLOCK_M, K] and B-tile [K, BLOCK_N]  (no masks needed)
    a = tl.load(A_ptr + offs_m[:, None] * K + offs_k[None, :])
    b = tl.load(B_ptr + offs_k[:, None] * N + offs_n[None, :])

    # Single dot product (K fully covered) + bias + residual fused
    acc = tl.dot(a, b, allow_tf32=True, out_dtype=tl.float32)

    bias_vals = tl.load(bias_ptr + offs_m)
    acc += bias_vals[:, None].to(tl.float32)

    r_vals = tl.load(R_ptr + offs_m[:, None] * N + offs_n[None, :])
    acc += r_vals.to(tl.float32)

    tl.store(out_ptr + offs_m[:, None] * N + offs_n[None, :],
             acc.to(out_ptr.dtype.element_ty))


# ---------------------------------------------------------------------------
# Replacement function
# ---------------------------------------------------------------------------
@torch.fx.wrap
def fused_conv_dropout_add(in_0, in_1, in_2, in_3):
    """
    in_0 : bias     [C_out]
    in_1 : weight   [C_out, C_in, 1, 1]
    in_2 : residual [N, C_out, H, W]
    in_3 : input    [N, C_in,  H, W]
    """
    N_batch, C_in, H, W = in_3.shape
    C_out = in_1.shape[0]
    HW    = H * W
    NHW   = N_batch * HW

    # Zero-copy 2-D views (correct for N_batch=1, contiguous NCHW)
    A = in_1.reshape(C_out, C_in)    # [M=128, K=256]
    B = in_3.reshape(C_in,  NHW)     # [K=256, N=1024]
    R = in_2.reshape(C_out, NHW)     # [M=128, N=1024]

    out_2d = torch.empty((C_out, NHW), dtype=in_3.dtype, device=in_3.device)

    grid = lambda meta: (
        triton.cdiv(C_out, meta['BLOCK_M']),
        triton.cdiv(NHW,   meta['BLOCK_N']),
    )

    fused_gemm_nomask_kernel[grid](
        A, B, in_0, R, out_2d,
        NHW,          # N (runtime stride parameter)
        M=C_out,      # constexpr = 128
        K=C_in,       # constexpr = 256
    )

    return out_2d.reshape(N_batch, C_out, H, W)


def replacement_func():
    return fused_conv_dropout_add