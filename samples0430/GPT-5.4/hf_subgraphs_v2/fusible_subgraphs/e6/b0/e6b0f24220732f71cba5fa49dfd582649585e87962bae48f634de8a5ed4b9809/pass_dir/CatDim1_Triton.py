import torch
import triton
import triton.language as tl


def pattern(in_2, in_3):
    tmp_0 = torch.cat((in_2, in_3), 1)
    return tmp_0


def replacement_args(in_2, in_3):
    return (in_2, in_3)


@triton.jit
def batched_copy_kernel(
    src_ptr,
    dst_ptr,
    total_elems,
    src_batch_stride,
    dst_batch_stride,
    batch_chunk_elems,
    dst_inner_base,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < total_elems

    batch = offs // batch_chunk_elems
    inner = offs % batch_chunk_elems

    src = tl.load(src_ptr + batch * src_batch_stride + inner, mask=mask)
    tl.store(dst_ptr + batch * dst_batch_stride + dst_inner_base + inner, src, mask=mask)


@torch.fx.wrap
def fused_cat_dim1(in_2, in_3):
    n = in_2.shape[0]
    c0 = in_2.shape[1]
    c1 = in_3.shape[1]
    h = in_2.shape[2]
    w = in_2.shape[3]

    out = torch.empty((n, c0 + c1, h, w), device=in_2.device, dtype=in_2.dtype)

    elems0 = in_2.numel()
    elems1 = in_3.numel()
    batch_chunk0 = c0 * h * w
    batch_chunk1 = c1 * h * w
    dst_batch_stride = (c0 + c1) * h * w

    block_size = 1024
    num_warps = 8

    batched_copy_kernel[(triton.cdiv(elems0, block_size),)](
        in_2,
        out,
        elems0,
        batch_chunk0,
        dst_batch_stride,
        batch_chunk0,
        0,
        BLOCK_SIZE=block_size,
        num_warps=num_warps,
    )

    batched_copy_kernel[(triton.cdiv(elems1, block_size),)](
        in_3,
        out,
        elems1,
        batch_chunk1,
        dst_batch_stride,
        batch_chunk1,
        batch_chunk0,
        BLOCK_SIZE=block_size,
        num_warps=num_warps,
    )

    return out


def replacement_func():
    return fused_cat_dim1