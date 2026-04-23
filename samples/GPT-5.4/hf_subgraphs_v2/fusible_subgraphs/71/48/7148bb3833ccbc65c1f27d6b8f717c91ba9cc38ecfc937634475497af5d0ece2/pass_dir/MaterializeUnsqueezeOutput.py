import torch
import triton
import triton.language as tl


def pattern(in_1):
    tmp_9 = in_1.unsqueeze(-1)
    tmp_10 = tmp_9.unsqueeze(-1)
    return tmp_10


def replacement_args(in_1):
    return (in_1,)


@triton.jit
def unsqueeze2_copy_kernel(
    src_ptr,
    out_ptr,
    C,
    src_stride_c,
    out_stride_c,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < C
    vals = tl.load(src_ptr + offs * src_stride_c, mask=mask, other=0.0)
    tl.store(out_ptr + offs * out_stride_c, vals, mask=mask)


@torch.fx.wrap
def materialize_unsqueeze_output(in_1):
    C = in_1.shape[0]
    out = torch.empty((C, 1, 1), device=in_1.device, dtype=in_1.dtype)
    src_stride_c = in_1.stride()[0]
    out_stride_c = out.stride()[0]
    BLOCK_SIZE = 64
    unsqueeze2_copy_kernel[(triton.cdiv(C, BLOCK_SIZE),)](
        in_1,
        out,
        C,
        src_stride_c,
        out_stride_c,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return out


def replacement_func():
    return materialize_unsqueeze_output