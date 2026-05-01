import torch
import triton
import triton.language as tl


# Match the fused mul+add computation that produces the [B,K,2,C] result.
# unbind and permute are zero-copy view ops downstream — no extra kernel needed.
# Fusing mul+add eliminates the intermediate [B,K,2,C] mul result and halves
# memory traffic vs. two separate PyTorch kernels.
def pattern(in_0, in_1, in_2):
    tmp_1 = in_2 * in_1
    tmp_2 = tmp_1 + in_0
    return tmp_2


def replacement_args(in_0, in_1, in_2):
    return (in_0, in_1, in_2)


# One Triton program per (b,k) pair.
# K=17 and C=128 are constant across all graph variants → use as tl.constexpr.
@triton.jit
def _fused_mul_add_kernel(
    in0_ptr,   # [2, C]        – bias, contiguous
    in1_ptr,   # [1, 1, 2, C]  – weight, contiguous
    in2_ptr,   # [B, K, 1, C]  – features (from unsqueeze)
    out_ptr,   # [B, K, 2, C]  – output, contiguous
    B,
    K: tl.constexpr,   # always 17
    C: tl.constexpr,   # always 128
):
    pid = tl.program_id(0)
    b = pid // K
    k = pid % K

    c = tl.arange(0, C)

    # Load in_2[b, k, 0, :] once — shared for both j=0 and j=1
    x = tl.load(in2_ptr + (b * K + k) * C + c)

    # j = 0
    w0    = tl.load(in1_ptr + c)
    bias0 = tl.load(in0_ptr + c)
    v0 = x * w0 + bias0

    # j = 1
    w1    = tl.load(in1_ptr + C + c)
    bias1 = tl.load(in0_ptr + C + c)
    v1 = x * w1 + bias1

    # Store out[b, k, 0, :] and out[b, k, 1, :] (both contiguous)
    base = (b * K + k) * 2 * C
    tl.store(out_ptr + base     + c, v0)
    tl.store(out_ptr + base + C + c, v1)


@torch.fx.wrap
def fused_mul_add(in_0, in_1, in_2):
    B = in_2.shape[0]   # only B varies; K=17 and C=128 are always constant

    out = torch.empty((B, 17, 2, 128), dtype=in_2.dtype, device=in_2.device)

    _fused_mul_add_kernel[(B * 17,)](
        in_0, in_1, in_2,
        out,
        B,
        K=17,
        C=128,
        num_warps=4,
    )

    return out


def replacement_func():
    return fused_mul_add