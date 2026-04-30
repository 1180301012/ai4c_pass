import torch
import triton
import triton.language as tl


def pattern(in_0, in_1, in_2):
    tmp_1 = in_1.norm(p=2, dim=-1, keepdim=True)
    tmp_2 = in_1 / tmp_1
    tmp_3 = in_2.norm(p=2, dim=-1, keepdim=True)
    tmp_4 = in_2 / tmp_3
    tmp_5 = in_0.exp()
    tmp_6 = tmp_5 * tmp_4
    return (tmp_6, tmp_4, tmp_2)


def replacement_args(in_0, in_1, in_2):
    return (in_0, in_1, in_2)


@triton.jit
def fused_two_norms_exp_mul_kernel(
    in0_ptr,
    in1_ptr,
    in2_ptr,
    out6_ptr,
    out4_ptr,
    out2_ptr,
    n_cols,
    stride1_last,
    stride2_last,
    BLOCK_SIZE: tl.constexpr,
):
    offs = tl.arange(0, BLOCK_SIZE)
    mask = offs < n_cols

    x1_raw = tl.load(in1_ptr + offs * stride1_last, mask=mask, other=0)
    x2_raw = tl.load(in2_ptr + offs * stride2_last, mask=mask, other=0)

    x1 = x1_raw.to(tl.float32)
    x2 = x2_raw.to(tl.float32)

    norm1 = tl.sqrt(tl.sum(x1 * x1, axis=0))
    norm2 = tl.sqrt(tl.sum(x2 * x2, axis=0))

    y2_fp32 = x1 / norm1
    y4_fp32 = x2 / norm2

    scale = tl.exp(tl.load(in0_ptr + 0).to(tl.float32))
    y6_fp32 = y4_fp32 * scale

    tl.store(out2_ptr + offs, y2_fp32.to(x1_raw.dtype), mask=mask)
    tl.store(out4_ptr + offs, y4_fp32.to(x2_raw.dtype), mask=mask)
    tl.store(out6_ptr + offs, y6_fp32.to(x2_raw.dtype), mask=mask)


@torch.fx.wrap
def fused_two_norms_exp_mul(in_0, in_1, in_2):
    n_cols = in_1.shape[-1]

    out2 = torch.empty_like(in_1)
    out4 = torch.empty_like(in_2)
    out6 = torch.empty_like(in_2)

    grid = (1,)
    fused_two_norms_exp_mul_kernel[grid](
        in_0,
        in_1,
        in_2,
        out6,
        out4,
        out2,
        n_cols,
        in_1.stride(-1),
        in_2.stride(-1),
        BLOCK_SIZE=512,
        num_warps=4,
        num_stages=1,
    )

    return (out6, out4, out2)


def replacement_func():
    return fused_two_norms_exp_mul