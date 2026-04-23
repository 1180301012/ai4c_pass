import torch
import triton
import triton.language as tl


OUT_FEATURES = 2


def pattern(in_0, in_1, in_2):
    linear = torch.nn.functional.linear(in_2, in_1, in_0)
    return linear


def replacement_args(in_0, in_1, in_2):
    return (in_0, in_1, in_2)


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 16, "BLOCK_K": 64}, num_warps=2, num_stages=2),
        triton.Config({"BLOCK_M": 32, "BLOCK_K": 64}, num_warps=2, num_stages=2),
        triton.Config({"BLOCK_M": 64, "BLOCK_K": 64}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 64, "BLOCK_K": 128}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 128, "BLOCK_K": 64}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 128, "BLOCK_K": 128}, num_warps=8, num_stages=2),
    ],
    key=["M"],
)
@triton.jit
def _linear2_kernel(
    bias_ptr,
    weight_ptr,
    x_ptr,
    out_ptr,
    M,
    weight_s0,
    weight_s1,
    x_s0,
    x_s1,
    out_s0,
    out_s1,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    m_mask = offs_m < M

    acc0 = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc1 = tl.zeros([BLOCK_M], dtype=tl.float32)

    for k in tl.static_range(0, 448, BLOCK_K):
        offs_k = k + tl.arange(0, BLOCK_K)
        k_mask = offs_k < 448
        x = tl.load(
            x_ptr + offs_m[:, None] * x_s0 + offs_k[None, :] * x_s1,
            mask=m_mask[:, None] & k_mask[None, :],
            other=0.0,
        ).to(tl.float32)
        w0 = tl.load(weight_ptr + 0 * weight_s0 + offs_k * weight_s1, mask=k_mask, other=0.0).to(tl.float32)
        w1 = tl.load(weight_ptr + 1 * weight_s0 + offs_k * weight_s1, mask=k_mask, other=0.0).to(tl.float32)
        acc0 += tl.sum(x * w0[None, :], axis=1)
        acc1 += tl.sum(x * w1[None, :], axis=1)

    b0 = tl.load(bias_ptr + 0).to(tl.float32)
    b1 = tl.load(bias_ptr + 1).to(tl.float32)
    acc0 += b0
    acc1 += b1

    tl.store(out_ptr + offs_m * out_s0 + 0 * out_s1, acc0, mask=m_mask)
    tl.store(out_ptr + offs_m * out_s0 + 1 * out_s1, acc1, mask=m_mask)


@torch.fx.wrap
def fused_linear_mean(in_0, in_1, in_2):
    M = in_2.shape[0]
    out = torch.empty((M, OUT_FEATURES), device=in_2.device, dtype=in_2.dtype)
    grid = lambda meta: (triton.cdiv(M, meta["BLOCK_M"]),)
    _linear2_kernel[grid](
        in_0,
        in_1,
        in_2,
        out,
        M,
        in_1.stride(0),
        in_1.stride(1),
        in_2.stride(0),
        in_2.stride(1),
        out.stride(0),
        out.stride(1),
    )
    return out


def replacement_func():
    return fused_linear_mean