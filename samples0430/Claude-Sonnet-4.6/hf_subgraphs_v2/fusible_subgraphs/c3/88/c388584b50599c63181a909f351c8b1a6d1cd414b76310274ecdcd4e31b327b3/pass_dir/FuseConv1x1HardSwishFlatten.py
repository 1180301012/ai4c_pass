import torch
import triton
import triton.language as tl


# ── Pattern ────────────────────────────────────────────────────────────────────
def pattern(in_0, in_1, in_2):
    """1×1 conv2d → hardswish(inplace) → flatten(1,-1)"""
    conv2d = torch.conv2d(in_2, in_1, in_0, (1, 1), (0, 0), (1, 1), 1)
    tmp_3  = torch.nn.functional.hardswish(conv2d, True)
    tmp_4  = tmp_3.flatten(1, -1)
    return tmp_4


def replacement_args(in_0, in_1, in_2):
    return (in_0, in_1, in_2)


# ── Triton GEMM+HardSwish kernel ───────────────────────────────────────────────
#
# Computes C = hardswish(A @ B.T + bias)
#   A : [M, K]  row-major (in_2 as 2-D)
#   B : [N, K]  row-major (in_1 as 2-D)
#   C : [M, N]  (already "flattened")
#
# B is loaded row-by-row ([BLOCK_N, BLOCK_K], k is fast → coalesced), then
# tl.trans(b) → [BLOCK_K, BLOCK_N] for tl.dot, avoiding shared-memory
# bank conflicts present in the old non-coalesced load approach.
#
# Only 4 configs so the autotuner gets ≈6 trials each within 25 warmup
# iterations, yielding reliable config selection.
#
@triton.autotune(
    configs=[
        # 40 CTAs for M=32 (2 M-blocks × 20 N-blocks) → 71% SM utilization
        triton.Config({'BLOCK_M': 16, 'BLOCK_N':  64, 'BLOCK_K': 64,
                       'num_warps': 4, 'num_stages': 4}),
        # 10 CTAs for M=32 (1 M-block × 10 N-blocks) → larger tiles
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 128, 'BLOCK_K': 64,
                       'num_warps': 4, 'num_stages': 4}),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def _fused_gemm_hardswish_kernel(
    a_ptr, b_ptr, bias_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bn, stride_bk,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    # A [BLOCK_M, BLOCK_K] – k is fast → coalesced
    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    # B [BLOCK_N, BLOCK_K] – k is fast → coalesced; transposed in tl.dot
    b_ptrs = b_ptr + offs_n[:, None] * stride_bn + offs_k[None, :] * stride_bk

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # K=960 ÷ {32, 64} is exact → no K-boundary masking
    for _ in range(0, tl.cdiv(K, BLOCK_K)):
        a = tl.load(a_ptrs, mask=offs_m[:, None] < M, other=0.0)
        b = tl.load(b_ptrs, mask=offs_n[:, None] < N, other=0.0)
        acc = tl.dot(a, tl.trans(b), acc)   # fused MAC into accumulator
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    # Bias add
    bias = tl.load(bias_ptr + offs_n, mask=offs_n < N, other=0.0)
    acc  = acc + bias[None, :].to(tl.float32)

    # HardSwish: x * clamp(x+3, 0, 6) / 6
    relu6 = tl.minimum(tl.maximum(acc + 3.0, 0.0), 6.0)
    out   = acc * relu6 * (1.0 / 6.0)

    # Store – Triton casts float32 → destination dtype automatically
    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(c_ptrs, out, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


# ── PyTorch wrapper ─────────────────────────────────────────────────────────────
@torch.fx.wrap
def fused_conv1x1_hardswish_flatten(in_0, in_1, in_2):
    M = in_2.shape[0]    # batch (1 or 32)
    K = in_2.shape[1]    # C_in  = 960
    N = in_1.shape[0]    # C_out = 1280

    out  = torch.empty((M, N), dtype=in_2.dtype, device=in_2.device)
    grid = lambda meta: (triton.cdiv(M, meta['BLOCK_M']),
                         triton.cdiv(N, meta['BLOCK_N']))

    _fused_gemm_hardswish_kernel[grid](
        in_2, in_1, in_0, out,
        M, N, K,
        in_2.stride(0), in_2.stride(1),
        in_1.stride(0), in_1.stride(1),
        out.stride(0),  out.stride(1),
    )
    return out


def replacement_func():
    return fused_conv1x1_hardswish_flatten