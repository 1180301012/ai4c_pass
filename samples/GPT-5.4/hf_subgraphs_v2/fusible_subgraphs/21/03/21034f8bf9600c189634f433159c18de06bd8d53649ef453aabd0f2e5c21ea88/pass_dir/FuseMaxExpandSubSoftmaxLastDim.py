import torch
import triton
import triton.language as tl


# Pattern matching function
# Mirrors the source graph exactly for the softmax-producing subgraph.
def pattern(in_0):
    tmp_0 = torch.max(in_0, -1, keepdim=True)
    tmp_1 = tmp_0[0]
    tmp_2 = tmp_1.expand_as(in_0)
    tmp_3 = tmp_2 - in_0
    tmp_4 = torch.nn.functional.softmax(tmp_3, dim=-1)
    return tmp_4


# Argument extraction function
def replacement_args(in_0):
    return (in_0,)


@triton.jit
def fused_max_sub_softmax_lastdim_kernel(
    x_ptr,
    out_ptr,
    dim0,
    dim1,
    dim2,
    x_stride0,
    x_stride1,
    x_stride2,
    out_stride0,
    out_stride1,
    out_stride2,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)

    row1 = pid % dim1
    row0 = pid // dim1

    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < dim2

    x_row_base = row0 * x_stride0 + row1 * x_stride1
    out_row_base = row0 * out_stride0 + row1 * out_stride1
    x_offsets = x_row_base + cols * x_stride2
    out_offsets = out_row_base + cols * out_stride2

    x_for_max = tl.load(x_ptr + x_offsets, mask=mask, other=-float("inf")).to(tl.float32)
    row_max = tl.max(x_for_max, axis=0)

    x_vals = tl.load(x_ptr + x_offsets, mask=mask, other=0.0).to(tl.float32)
    shifted = row_max - x_vals
    shifted = tl.where(mask, shifted, -float("inf"))
    numer = tl.exp(shifted)
    denom = tl.sum(numer, axis=0)
    out = numer / denom

    tl.store(out_ptr + out_offsets, out, mask=mask)


@torch.fx.wrap
def fused_max_sub_softmax_lastdim(in_0):
    dim0 = in_0.shape[0]
    dim1 = in_0.shape[1]
    dim2 = in_0.shape[2]

    out = torch.empty_like(in_0)

    # These benchmark graphs all use dim2 == 512. Keep a small dynamic ladder
    # for robustness without introducing extra kernels or complexity.
    if dim2 <= 128:
        block_size = 128
        num_warps = 4
    elif dim2 <= 256:
        block_size = 256
        num_warps = 4
    elif dim2 <= 512:
        block_size = 512
        num_warps = 8
    else:
        block_size = 1024
        num_warps = 8

    grid = (dim0 * dim1,)
    fused_max_sub_softmax_lastdim_kernel[grid](
        in_0,
        out,
        dim0,
        dim1,
        dim2,
        in_0.stride(0),
        in_0.stride(1),
        in_0.stride(2),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        BLOCK_SIZE=block_size,
        num_warps=num_warps,
        num_stages=2,
    )
    return out


# Replacement function (NO arguments, returns function reference)
def replacement_func():
    return fused_max_sub_softmax_lastdim