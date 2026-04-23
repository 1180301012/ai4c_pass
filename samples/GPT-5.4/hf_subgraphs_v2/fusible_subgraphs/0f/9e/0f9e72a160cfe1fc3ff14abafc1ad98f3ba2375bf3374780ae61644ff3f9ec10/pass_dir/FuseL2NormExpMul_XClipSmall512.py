import torch
import triton
import triton.language as tl


# Pattern matching function
# Must mirror model.py exactly, including keyword arguments.
def pattern(in_0, in_1, in_2):
    tmp_1 = in_1.norm(p=2, dim=-1, keepdim=True)
    tmp_2 = in_1 / tmp_1
    tmp_3 = in_2.norm(p=2, dim=-1, keepdim=True)
    tmp_4 = in_2 / tmp_3
    tmp_5 = in_0.exp()
    tmp_6 = tmp_5 * tmp_4
    return (tmp_6, tmp_4, tmp_2)


# Argument extraction function
def replacement_args(in_0, in_1, in_2):
    return (in_0, in_1, in_2)


@triton.jit
def _fused_xclip_norm_scale_kernel(
    in1_ptr,
    in2_ptr,
    scalar_ptr,
    out6_ptr,
    out4_ptr,
    out2_ptr,
    stride_in1_col,
    stride_in2_col,
    stride_out6_col,
    stride_out4_col,
    stride_out2_col,
    N1,
    N2,
    BLOCK: tl.constexpr,
):
    offs = tl.arange(0, BLOCK)
    mask1 = offs < N1
    mask2 = offs < N2

    x1 = tl.load(in1_ptr + offs * stride_in1_col, mask=mask1, other=0.0).to(tl.float32)
    sumsq1 = tl.sum(x1 * x1, axis=0)
    norm1 = tl.sqrt(sumsq1)
    y1 = x1 / norm1

    x2 = tl.load(in2_ptr + offs * stride_in2_col, mask=mask2, other=0.0).to(tl.float32)
    sumsq2 = tl.sum(x2 * x2, axis=0)
    norm2 = tl.sqrt(sumsq2)
    y2 = x2 / norm2

    s = tl.load(scalar_ptr).to(tl.float32)
    scale = tl.exp(s)
    z2 = y2 * scale

    tl.store(out2_ptr + offs * stride_out2_col, y1, mask=mask1)
    tl.store(out4_ptr + offs * stride_out4_col, y2, mask=mask2)
    tl.store(out6_ptr + offs * stride_out6_col, z2, mask=mask2)


@torch.fx.wrap
def fused_xclip_norm_scale(in_0, in_1, in_2):
    out2 = torch.empty_like(in_1)
    out4 = torch.empty_like(in_2)
    out6 = torch.empty_like(in_2)

    n1 = int(in_1.shape[-1])
    n2 = int(in_2.shape[-1])
    block = triton.next_power_of_2(max(n1, n2))
    block = 512 if block < 512 else block

    _fused_xclip_norm_scale_kernel[(1,)](
        in_1,
        in_2,
        in_0,
        out6,
        out4,
        out2,
        in_1.stride(1),
        in_2.stride(2),
        out6.stride(2),
        out4.stride(2),
        out2.stride(1),
        n1,
        n2,
        BLOCK=block,
        num_warps=4,
    )

    return (out6, out4, out2)


# Replacement function (NO arguments, returns function reference)
def replacement_func():
    return fused_xclip_norm_scale