import operator

import torch
import triton
import triton.language as tl


# Pattern matching function
# Mirrors model.py exactly:
#   tmp_0 = in_0 / 8.0
#   tmp_0 += in_2
#   tmp_1 = tmp_0
#   tmp_2 = tmp_1 + in_1
#   return (tmp_2,)
def pattern(in_0, in_1, in_2):
    tmp_0 = in_0 / 8.0
    tmp_0 = operator.iadd(tmp_0, in_2)
    tmp_1 = tmp_0
    tmp_2 = tmp_1 + in_1
    return tmp_2


# Argument extraction function
def replacement_args(in_0, in_1, in_2):
    return (in_0, in_1, in_2)


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": 256}, num_warps=2, num_stages=1),
        triton.Config({"BLOCK_SIZE": 512}, num_warps=4, num_stages=1),
        triton.Config({"BLOCK_SIZE": 1024}, num_warps=4, num_stages=1),
    ],
    key=["n_elements"],
)
@triton.jit
def fused_div8_add_value_add_mask_kernel(
    x0_ptr,
    x1_ptr,
    x2_ptr,
    out_ptr,
    d0,
    d1,
    d2,
    d3,
    x0_s0,
    x0_s1,
    x0_s2,
    x0_s3,
    x1_s0,
    x1_s1,
    x1_s2,
    x1_s3,
    x2_s0,
    x2_s1,
    x2_s2,
    x2_s3,
    out_s0,
    out_s1,
    out_s2,
    out_s3,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    tmp = offsets
    i3 = tmp % d3
    tmp = tmp // d3
    i2 = tmp % d2
    tmp = tmp // d2
    i1 = tmp % d1
    i0 = tmp // d1

    x0_offsets = i0 * x0_s0 + i1 * x0_s1 + i2 * x0_s2 + i3 * x0_s3
    # x1 has shape [d0, 1, 1, d3]; singleton dimensions are broadcast.
    x1_offsets = i0 * x1_s0 + i3 * x1_s3
    x2_offsets = i0 * x2_s0 + i1 * x2_s1 + i2 * x2_s2 + i3 * x2_s3
    out_offsets = i0 * out_s0 + i1 * out_s1 + i2 * out_s2 + i3 * out_s3

    x0 = tl.load(x0_ptr + x0_offsets, mask=mask, other=0.0)
    x1 = tl.load(x1_ptr + x1_offsets, mask=mask, other=0.0)
    x2 = tl.load(x2_ptr + x2_offsets, mask=mask, other=0.0)

    out = x0 / 8.0 + x2 + x1

    tl.store(out_ptr + out_offsets, out, mask=mask)


@torch.fx.wrap
def fused_div8_add_value_add_mask(in_0, in_1, in_2):
    out = torch.empty_like(in_0)

    n_elements = out.numel()
    d0 = out.size(0)
    d1 = out.size(1)
    d2 = out.size(2)
    d3 = out.size(3)

    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)

    fused_div8_add_value_add_mask_kernel[grid](
        in_0,
        in_1,
        in_2,
        out,
        d0,
        d1,
        d2,
        d3,
        in_0.stride(0),
        in_0.stride(1),
        in_0.stride(2),
        in_0.stride(3),
        in_1.stride(0),
        in_1.stride(1),
        in_1.stride(2),
        in_1.stride(3),
        in_2.stride(0),
        in_2.stride(1),
        in_2.stride(2),
        in_2.stride(3),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        out.stride(3),
        n_elements=n_elements,
    )

    return out


# Replacement function (NO arguments, returns function reference)
def replacement_func():
    return fused_div8_add_value_add_mask