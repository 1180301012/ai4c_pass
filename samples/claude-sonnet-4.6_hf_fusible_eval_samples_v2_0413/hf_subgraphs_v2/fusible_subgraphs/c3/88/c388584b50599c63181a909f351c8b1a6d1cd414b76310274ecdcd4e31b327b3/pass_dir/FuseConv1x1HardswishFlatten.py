import torch
import triton
import triton.language as tl


def pattern(in_0, in_1, in_2):
    """
    Match: 1x1 conv2d + hardswish (inplace) + flatten(1,-1)
    in_0: bias  [N]
    in_1: weight [N, K, 1, 1]
    in_2: input  [B, K, 1, 1]
    """
    conv2d = torch.conv2d(in_2, in_1, in_0, (1, 1), (0, 0), (1, 1), 1)
    tmp_3 = torch.nn.functional.hardswish(conv2d, True)
    tmp_4 = tmp_3.flatten(1, -1)
    return tmp_4


def replacement_args(in_0, in_1, in_2):
    return (in_0, in_1, in_2)


# -----------------------------------------------------------------------
#  Fused 1×1-conv + hardswish GEMM kernel.
#
#  Key optimisation: keep input & weight in their NATIVE dtype (fp16/bf16/fp32)
#  and let tl.dot accumulate into fp32 via out_dtype=tl.float32.
#  This allows Tensor Cores for fp16 / bf16 (2-8× faster than fp32 FMA).
#
#  Pointer-advancement style (like the Triton tutorial matmul) avoids
#  recomputing offsets every K-iteration.
# -----------------------------------------------------------------------
@triton.jit
def _fused_gemm_hs_kernel(
    input_ptr,   # [B, K]
    weight_ptr,  # [N, K]
    bias_ptr,    # [N]
    output_ptr,  # [B, N]
    B, N, K,
    stride_am, stride_ak,
    stride_bn, stride_bk,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    HAS_M_MASK: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    a_ptrs = input_ptr  + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = weight_ptr + offs_n[:, None] * stride_bn + offs_k[None, :] * stride_bk

    # fp32 accumulator for numerical accuracy
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for _ in range(0, tl.cdiv(K, BLOCK_K)):
        # Keep NATIVE dtype so tl.dot can use Tensor Cores (fp16/bf16).
        if HAS_M_MASK:
            a = tl.load(a_ptrs, mask=offs_m[:, None] < B, other=0.0)
        else:
            a = tl.load(a_ptrs)

        b = tl.load(b_ptrs, mask=offs_n[:, None] < N, other=0.0)

        # out_dtype=float32: accumulate in fp32 even if inputs are fp16/bf16
        acc = tl.dot(a, tl.trans(b), acc, out_dtype=tl.float32)

        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    # Bias add (convert to fp32)
    bias = tl.load(bias_ptr + offs_n, mask=offs_n < N, other=0.0)
    acc += bias[None, :].to(tl.float32)

    # Hardswish: x * clamp(x+3, 0, 6) / 6
    out = acc * tl.minimum(tl.maximum(acc + 3.0, 0.0), 6.0) * (1.0 / 6.0)

    # Store (Triton auto-converts fp32 → fp16/bf16 via pointer dtype)
    mask_out = offs_n[None, :] < N
    if HAS_M_MASK:
        mask_out = mask_out & (offs_m[:, None] < B)
    tl.store(output_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn,
             out, mask=mask_out)


@torch.fx.wrap
def fused_conv1x1_hardswish(bias, weight, input_tensor):
    """
    Fused 1×1-conv + hardswish + flatten.
    bias:         [N]
    weight:       [N, K, 1, 1]
    input_tensor: [B, K, 1, 1]
    Returns:      [B, N]
    """
    B = input_tensor.shape[0]
    N = weight.shape[0]
    K = weight.shape[1]

    output = torch.empty((B, N), dtype=input_tensor.dtype, device=input_tensor.device)

    # BLOCK_M=32 covers B=32 in one tile (no wasted rows, best weight reuse).
    # BLOCK_N=64 → grid=(1,20) = 20 CTAs for N=1280; all run in parallel on A30.
    # num_stages=4: prefetch 3 K-blocks ahead to hide L2/DRAM latency.
    BLOCK_M = 32
    BLOCK_N = 64
    BLOCK_K = 64

    grid = (triton.cdiv(B, BLOCK_M), triton.cdiv(N, BLOCK_N))
    has_m_mask = (B % BLOCK_M != 0)

    _fused_gemm_hs_kernel[grid](
        input_tensor, weight, bias, output,
        B, N, K,
        input_tensor.stride(0), input_tensor.stride(1),
        weight.stride(0),        weight.stride(1),
        output.stride(0),        output.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        HAS_M_MASK=has_m_mask,
        num_stages=3, num_warps=4,
    )

    return output


def replacement_func():
    return fused_conv1x1_hardswish