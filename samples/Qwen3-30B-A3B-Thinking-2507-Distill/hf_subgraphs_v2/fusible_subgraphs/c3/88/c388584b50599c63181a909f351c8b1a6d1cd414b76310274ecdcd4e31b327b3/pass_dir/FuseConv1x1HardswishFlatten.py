import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Fused 1×1 Conv2d + HardSwish  (2-D grid: M × N tiles, inner K-loop)
#
# The 3-D grid (M×N×K) is WRONG: each CTA must accumulate over ALL K blocks
# to produce a correct dot product.  A 2-D grid with an inner K-loop is the
# correct approach.  Each CTA handles one (M_tile, N_tile) output tile and
# loops over all K blocks with tl.dot (native dtype → tensor cores).
# ---------------------------------------------------------------------------

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 64,  'BLOCK_K': 64},  num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 128, 'BLOCK_K': 64},  num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64,  'BLOCK_K': 64},  num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 128, 'BLOCK_K': 64},  num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 64},  num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 64,  'BLOCK_K': 32},  num_stages=5, num_warps=4),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 128, 'BLOCK_K': 32},  num_stages=5, num_warps=8),
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 128, 'BLOCK_K': 32},  num_stages=5, num_warps=4),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def _fused_1x1_hardswish(
    input_ptr,    # [M, K]
    weight_ptr,   # [N, K]
    bias_ptr,     # [N]
    output_ptr,   # [M, N]  (native dtype)
    M, N, K,
    stride_im, stride_ik,
    stride_wn, stride_wk,
    IS_BF16: tl.constexpr,
    IS_FP16: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """
    2-D grid GEMM + HardSwish.
    pid_0 = M tile, pid_1 = N tile.
    Each CTA accumulates over ALL K blocks (inner loop) and fires tensor cores.
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k_start in range(0, tl.cdiv(K, BLOCK_K)):
        offs_k = k_start * BLOCK_K + tl.arange(0, BLOCK_K)

        # Load input [BLOCK_M, BLOCK_K] in native dtype (BF16/FP16)
        a_ptrs = input_ptr + offs_m[:, None] * stride_im + offs_k[None, :] * stride_ik
        a_mask = (offs_m[:, None] < M) & (offs_k[None, :] < K)
        a = tl.load(a_ptrs, mask=a_mask, other=0.0)

        # Load weight [BLOCK_N, BLOCK_K] in native dtype
        b_ptrs = weight_ptr + offs_n[:, None] * stride_wn + offs_k[None, :] * stride_wk
        b_mask = (offs_n[:, None] < N) & (offs_k[None, :] < K)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)

        # Native-dtype tl.dot → tensor-core WMMA instructions on Ampere
        acc = tl.dot(a, tl.trans(b), acc)

    # Add bias
    bias = tl.load(bias_ptr + offs_n, mask=offs_n < N, other=0.0).to(tl.float32)
    acc  = acc + bias[None, :]

    # HardSwish: x * clamp(x + 3, 0, 6) / 6
    tmp = acc + 3.0
    acc = acc * tl.minimum(tl.maximum(tmp, 0.0), 6.0) * (1.0 / 6.0)

    # Store in native output dtype
    out_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    out_ptrs = output_ptr + offs_m[:, None] * N + offs_n[None, :]
    if IS_BF16:
        tl.store(out_ptrs, acc.to(tl.bfloat16), mask=out_mask)
    elif IS_FP16:
        tl.store(out_ptrs, acc.to(tl.float16), mask=out_mask)
    else:
        tl.store(out_ptrs, acc, mask=out_mask)


@torch.fx.wrap
def fused_conv1x1_hardswish_flatten(bias, weight, input_tensor):
    """
    Fused 1×1 Conv2d + HardSwish + Flatten.
    Args match replacement_args: (in_0=bias, in_1=weight, in_2=input_tensor)
    """
    M = input_tensor.shape[0]   # batch
    K = input_tensor.shape[1]   # C_in
    N = weight.shape[0]          # C_out

    is_bf16 = (input_tensor.dtype == torch.bfloat16)
    is_fp16 = (input_tensor.dtype == torch.float16)

    output = torch.empty((M, N), device=input_tensor.device, dtype=input_tensor.dtype)

    grid = lambda meta: (
        triton.cdiv(M, meta['BLOCK_M']),
        triton.cdiv(N, meta['BLOCK_N']),
    )

    _fused_1x1_hardswish[grid](
        input_tensor, weight, bias, output,
        M, N, K,
        input_tensor.stride(0), input_tensor.stride(1),
        weight.stride(0),       weight.stride(1),
        IS_BF16=is_bf16,
        IS_FP16=is_fp16,
    )

    return output


# ---------------------------------------------------------------------------
# Pattern / replacement wiring
# ---------------------------------------------------------------------------

def pattern(in_0, in_1, in_2):
    conv2d = torch.conv2d(in_2, in_1, in_0, (1, 1), (0, 0), (1, 1), 1)
    tmp_3  = torch.nn.functional.hardswish(conv2d, True)
    tmp_4  = tmp_3.flatten(1, -1)
    return tmp_4


def replacement_args(in_0, in_1, in_2):
    return (in_0, in_1, in_2)


def replacement_func():
    return fused_conv1x1_hardswish_flatten