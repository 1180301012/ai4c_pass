import torch
import triton
import triton.language as tl


# ─────────────────────────────────────────────────────────────────────────────
# Triton kernel: fused 1×1 conv (GEMM) + bias + residual add
#
# Memory layout (NCHW):
#   input  [N_batch, C_in,  H, W]  → viewed as [M=N*H*W, K=C_in]
#     stride_m = 1,   stride_k = H*W
#   weight [C_out, C_in, 1, 1]     → viewed as [N=C_out, K=C_in]
#     stride_n = C_in=K,  stride_k = 1
#   residual / output [N_batch, C_out, H, W] → viewed as [M, N]
#     stride_m = 1,   stride_n = H*W
# ─────────────────────────────────────────────────────────────────────────────
@triton.autotune(
    configs=[
        # ── Small blocks → many CTAs → high GPU occupancy (best for A30 56 SMs) ──
        # (16,16): grid=(64,8)=512 CTAs → 9 waves
        triton.Config({'BLOCK_M': 16,  'BLOCK_N': 16,  'BLOCK_K': 32},  num_stages=5, num_warps=2),
        triton.Config({'BLOCK_M': 16,  'BLOCK_N': 16,  'BLOCK_K': 64},  num_stages=4, num_warps=2),
        triton.Config({'BLOCK_M': 16,  'BLOCK_N': 16,  'BLOCK_K': 128}, num_stages=3, num_warps=2),
        triton.Config({'BLOCK_M': 16,  'BLOCK_N': 16,  'BLOCK_K': 256}, num_stages=2, num_warps=2),
        # (32,16) / (16,32): grid=(32,8)/(64,4)=256 CTAs → 4.6 waves
        triton.Config({'BLOCK_M': 32,  'BLOCK_N': 16,  'BLOCK_K': 32},  num_stages=5, num_warps=2),
        triton.Config({'BLOCK_M': 32,  'BLOCK_N': 16,  'BLOCK_K': 64},  num_stages=4, num_warps=2),
        triton.Config({'BLOCK_M': 32,  'BLOCK_N': 16,  'BLOCK_K': 128}, num_stages=3, num_warps=2),
        triton.Config({'BLOCK_M': 32,  'BLOCK_N': 16,  'BLOCK_K': 256}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_M': 16,  'BLOCK_N': 32,  'BLOCK_K': 32},  num_stages=5, num_warps=2),
        triton.Config({'BLOCK_M': 16,  'BLOCK_N': 32,  'BLOCK_K': 64},  num_stages=4, num_warps=2),
        triton.Config({'BLOCK_M': 16,  'BLOCK_N': 32,  'BLOCK_K': 128}, num_stages=3, num_warps=2),
        triton.Config({'BLOCK_M': 16,  'BLOCK_N': 32,  'BLOCK_K': 256}, num_stages=2, num_warps=4),
        # (32,32): grid=(32,4)=128 CTAs → 2.3 waves
        triton.Config({'BLOCK_M': 32,  'BLOCK_N': 32,  'BLOCK_K': 32},  num_stages=5, num_warps=4),
        triton.Config({'BLOCK_M': 32,  'BLOCK_N': 32,  'BLOCK_K': 64},  num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 32,  'BLOCK_N': 32,  'BLOCK_K': 128}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 32,  'BLOCK_N': 32,  'BLOCK_K': 256}, num_stages=2, num_warps=4),
        # ── Medium blocks ────────────────────────────────────────────────────────
        # (64,32)/(32,64): grid=(16,4)/(32,2)=64 CTAs → ~1.1 wave
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 32,  'BLOCK_K': 32},  num_stages=5, num_warps=4),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 32,  'BLOCK_K': 64},  num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 32,  'BLOCK_K': 128}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 32,  'BLOCK_N': 64,  'BLOCK_K': 32},  num_stages=5, num_warps=4),
        triton.Config({'BLOCK_M': 32,  'BLOCK_N': 64,  'BLOCK_K': 64},  num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 32,  'BLOCK_N': 64,  'BLOCK_K': 128}, num_stages=3, num_warps=4),
        # (64,64): grid=(16,2)=32 CTAs
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 64,  'BLOCK_K': 32},  num_stages=5, num_warps=4),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 64,  'BLOCK_K': 64},  num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 64,  'BLOCK_K': 128}, num_stages=3, num_warps=4),
        # ── Large blocks ─────────────────────────────────────────────────────────
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64,  'BLOCK_K': 32},  num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64,  'BLOCK_K': 64},  num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 128, 'BLOCK_K': 32},  num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 128, 'BLOCK_K': 64},  num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32},  num_stages=4, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64},  num_stages=4, num_warps=8),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def _conv1x1_bias_add_kernel(
    input_ptr,    # [M, K] with strides [1, HW]
    weight_ptr,   # [N, K] with strides [K, 1]
    bias_ptr,     # [N]
    residual_ptr, # [M, N] with strides [1, HW]
    output_ptr,   # [M, N] with strides [1, HW]
    M, N, K,
    stride_ik,    # H*W — stride for channel dim in input
    stride_rn,    # H*W — stride for channel dim in residual/output
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k_start in range(0, K, BLOCK_K):
        offs_k = k_start + tl.arange(0, BLOCK_K)

        # ── Load input tile [BLOCK_M, BLOCK_K] ──────────────────────────────
        # input[m, k] = input_ptr + m*1 + k*stride_ik
        a_ptrs = input_ptr + offs_m[:, None] + offs_k[None, :] * stride_ik
        mask_a = (offs_m[:, None] < M) & (offs_k[None, :] < K)
        a = tl.load(a_ptrs, mask=mask_a, other=0.0).to(tl.float32)

        # ── Load weight tile [BLOCK_N, BLOCK_K] ─────────────────────────────
        # weight[n, k] = weight_ptr + n*K + k
        b_ptrs = weight_ptr + offs_n[:, None] * K + offs_k[None, :]
        mask_b = (offs_n[:, None] < N) & (offs_k[None, :] < K)
        b = tl.load(b_ptrs, mask=mask_b, other=0.0).to(tl.float32)

        # acc += A [BLOCK_M, BLOCK_K] @ B^T [BLOCK_K, BLOCK_N]
        acc += tl.dot(a, tl.trans(b))

    # ── Add bias ─────────────────────────────────────────────────────────────
    bias = tl.load(bias_ptr + offs_n, mask=(offs_n < N), other=0.0).to(tl.float32)
    acc += bias[None, :]

    # ── Load residual, add, and store ────────────────────────────────────────
    mask_out = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    res_ptrs = residual_ptr + offs_m[:, None] + offs_n[None, :] * stride_rn
    res = tl.load(res_ptrs, mask=mask_out, other=0.0)
    acc += res.to(tl.float32)

    out_ptrs = output_ptr + offs_m[:, None] + offs_n[None, :] * stride_rn
    tl.store(out_ptrs, acc.to(res.dtype), mask=mask_out)


# ─────────────────────────────────────────────────────────────────────────────
# Wrapper — called by the replacement framework
# ─────────────────────────────────────────────────────────────────────────────
@torch.fx.wrap
def conv1x1_bias_dropout_add(in_0, in_1, in_2, in_3):
    """
    in_0 : bias     [C_out]
    in_1 : weight   [C_out, C_in, 1, 1]
    in_2 : residual [N_batch, C_out, H, W]
    in_3 : input    [N_batch, C_in,  H, W]
    """
    N_batch, C_in, H, W = in_3.shape
    C_out = in_1.shape[0]

    M  = N_batch * H * W   # spatial total
    K  = C_in               # input channels
    N  = C_out              # output channels
    HW = H * W              # used as channel stride in NCHW layout

    output = torch.empty_like(in_2)

    grid = lambda meta: (
        triton.cdiv(M, meta['BLOCK_M']),
        triton.cdiv(N, meta['BLOCK_N']),
    )

    _conv1x1_bias_add_kernel[grid](
        in_3, in_1, in_0, in_2, output,
        M, N, K,
        HW,   # stride_ik
        HW,   # stride_rn
    )

    return output


# ─────────────────────────────────────────────────────────────────────────────
# Pattern / replacement interface
# ─────────────────────────────────────────────────────────────────────────────
def pattern(in_0, in_1, in_2, in_3):
    conv    = torch.conv2d(in_3, in_1, in_0, (1, 1), (0, 0), (1, 1), 1)
    dropped = torch.nn.functional.dropout(conv, 0.0, False, False)
    out     = dropped + in_2
    return out


def replacement_args(in_0, in_1, in_2, in_3):
    return (in_0, in_1, in_2, in_3)


def replacement_func():
    return conv1x1_bias_dropout_add