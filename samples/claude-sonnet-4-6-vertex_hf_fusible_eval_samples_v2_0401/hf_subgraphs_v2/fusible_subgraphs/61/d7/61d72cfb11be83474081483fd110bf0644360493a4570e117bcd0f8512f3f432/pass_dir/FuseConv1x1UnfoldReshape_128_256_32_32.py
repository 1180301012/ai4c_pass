import torch
import triton
import triton.language as tl


# ──────────────────────────────────────────────────────────────────────────────
# Pattern: 1x1 conv2d
# Shapes:
#   in_0 (weight) : [128, 256, 1, 1]
#   in_1 (input)  : [1,   256, 32, 32]
#   conv2d output : [1,   128, 32, 32]
# ──────────────────────────────────────────────────────────────────────────────

def pattern(in_0, in_1):
    conv2d = torch.conv2d(in_1, in_0, None, (1, 1), (0, 0), (1, 1), 1)
    return conv2d


def replacement_args(in_0, in_1):
    return (in_0, in_1)


# ──────────────────────────────────────────────────────────────────────────────
# Triton GEMM kernel:  weight[128,256] @ input[256,1024] → output[128,1024]
# Tile pid_m over M=128, pid_n over N=1024, inner loop over K=256.
# Uses native FP16/BF16 tensor cores via tl.dot (no upcast to FP32).
# ──────────────────────────────────────────────────────────────────────────────

@triton.jit
def _conv1x1_gemm_kernel(
    weight_ptr,
    input_ptr,
    output_ptr,
    IS_BF16: tl.constexpr,
    IS_FP16: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    C_OUT = 128
    C_IN  = 256
    HW    = 1024

    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    m_offs = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    n_offs = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    m_mask = m_offs < C_OUT
    n_mask = n_offs < HW

    acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)

    for k_start in range(0, C_IN, BLOCK_K):
        k_offs = k_start + tl.arange(0, BLOCK_K)
        k_mask = k_offs < C_IN

        w = tl.load(weight_ptr + m_offs[:, None] * C_IN + k_offs[None, :],
                    mask=m_mask[:, None] & k_mask[None, :], other=0.0)
        x = tl.load(input_ptr  + k_offs[:, None] * HW   + n_offs[None, :],
                    mask=k_mask[:, None] & n_mask[None, :], other=0.0)

        acc = tl.dot(w, x, acc)

    if IS_BF16:
        out = acc.to(tl.bfloat16)
    elif IS_FP16:
        out = acc.to(tl.float16)
    else:
        out = acc
    tl.store(output_ptr + m_offs[:, None] * HW + n_offs[None, :],
             out, mask=m_mask[:, None] & n_mask[None, :])


@torch.fx.wrap
def fused_conv_unfold_reshape(in_0, in_1):
    dtype  = in_1.dtype
    device = in_1.device
    output = torch.empty((1, 128, 32, 32), dtype=dtype, device=device)
    is_bf16 = (dtype == torch.bfloat16)
    is_fp16 = (dtype == torch.float16)
    BLOCK_M, BLOCK_N, BLOCK_K = 64, 128, 32
    grid = (triton.cdiv(128, BLOCK_M), triton.cdiv(1024, BLOCK_N))
    _conv1x1_gemm_kernel[grid](
        in_0, in_1, output,
        IS_BF16=is_bf16, IS_FP16=is_fp16,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        num_warps=8, num_stages=4,
    )
    return output


def replacement_func():
    return fused_conv_unfold_reshape