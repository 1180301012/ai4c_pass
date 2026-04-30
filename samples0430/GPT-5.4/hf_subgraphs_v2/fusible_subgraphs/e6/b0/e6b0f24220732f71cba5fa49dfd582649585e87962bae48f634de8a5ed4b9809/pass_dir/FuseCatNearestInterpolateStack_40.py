import torch
import triton
import triton.language as tl

OUT_H = 40
OUT_W = 40
OUT_HW = OUT_H * OUT_W
IN1_H = 20
IN1_W = 20
IN1_HW = IN1_H * IN1_W
C_OUT = 512
C_HALF = 256
PLANE_ELEMS_PER_BATCH = C_OUT * OUT_HW
HALF_PLANE_ELEMS_PER_BATCH = C_HALF * OUT_HW
IN1_ELEMS_PER_BATCH = C_OUT * IN1_HW


def pattern(in_0, in_1, in_2, in_3):
    tmp_0 = torch.cat((in_2, in_3), 1)
    tmp_2 = torch.nn.functional.interpolate(in_1, size=(40, 40), mode='nearest')
    tmp_3 = torch.stack([in_0, tmp_2, tmp_0])
    return tmp_3


def replacement_args(in_0, in_1, in_2, in_3):
    return (in_0, in_1, in_2, in_3)


@triton.jit
def batched_copy_kernel(
    src_ptr,
    dst_ptr,
    total_elems,
    base_offset,
    BLOCK_SIZE: tl.constexpr,
    SRC_BATCH_STRIDE: tl.constexpr,
    DST_BATCH_STRIDE: tl.constexpr,
    BATCH_CHUNK_ELEMS: tl.constexpr,
    DST_INNER_BASE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < total_elems

    batch = offs // BATCH_CHUNK_ELEMS
    inner = offs % BATCH_CHUNK_ELEMS

    src = tl.load(src_ptr + batch * SRC_BATCH_STRIDE + inner, mask=mask)
    tl.store(dst_ptr + base_offset + batch * DST_BATCH_STRIDE + DST_INNER_BASE + inner, src, mask=mask)


@triton.jit
def upsample2x_plane1_kernel(
    src_ptr,
    dst_ptr,
    total_src_elems,
    base_offset,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < total_src_elems

    nc = offs // IN1_HW
    hw = offs % IN1_HW
    ih = hw // IN1_W
    iw = hw % IN1_W

    dst_base = base_offset + nc * OUT_HW + ih * (2 * OUT_W) + iw * 2
    x = tl.load(src_ptr + offs, mask=mask)

    tl.store(dst_ptr + dst_base, x, mask=mask)
    tl.store(dst_ptr + dst_base + 1, x, mask=mask)
    tl.store(dst_ptr + dst_base + OUT_W, x, mask=mask)
    tl.store(dst_ptr + dst_base + OUT_W + 1, x, mask=mask)


@torch.fx.wrap
def fused_cat_nearest_interpolate_stack_40(in_0, in_1, in_2, in_3):
    n = in_0.shape[0]
    out = torch.empty((3, n, C_OUT, OUT_H, OUT_W), device=in_0.device, dtype=in_0.dtype)

    plane_elems = n * PLANE_ELEMS_PER_BATCH
    half_plane_elems = n * HALF_PLANE_ELEMS_PER_BATCH
    in1_total_elems = n * IN1_ELEMS_PER_BATCH

    block_size = 1024
    num_warps = 8

    batched_copy_kernel[(triton.cdiv(plane_elems, block_size),)](
        in_0,
        out,
        plane_elems,
        0,
        BLOCK_SIZE=block_size,
        SRC_BATCH_STRIDE=PLANE_ELEMS_PER_BATCH,
        DST_BATCH_STRIDE=PLANE_ELEMS_PER_BATCH,
        BATCH_CHUNK_ELEMS=PLANE_ELEMS_PER_BATCH,
        DST_INNER_BASE=0,
        num_warps=num_warps,
    )

    upsample2x_plane1_kernel[(triton.cdiv(in1_total_elems, block_size),)](
        in_1,
        out,
        in1_total_elems,
        plane_elems,
        BLOCK_SIZE=block_size,
        num_warps=num_warps,
    )

    plane2_base = 2 * plane_elems
    batched_copy_kernel[(triton.cdiv(half_plane_elems, block_size),)](
        in_2,
        out,
        half_plane_elems,
        plane2_base,
        BLOCK_SIZE=block_size,
        SRC_BATCH_STRIDE=HALF_PLANE_ELEMS_PER_BATCH,
        DST_BATCH_STRIDE=PLANE_ELEMS_PER_BATCH,
        BATCH_CHUNK_ELEMS=HALF_PLANE_ELEMS_PER_BATCH,
        DST_INNER_BASE=0,
        num_warps=num_warps,
    )

    batched_copy_kernel[(triton.cdiv(half_plane_elems, block_size),)](
        in_3,
        out,
        half_plane_elems,
        plane2_base,
        BLOCK_SIZE=block_size,
        SRC_BATCH_STRIDE=HALF_PLANE_ELEMS_PER_BATCH,
        DST_BATCH_STRIDE=PLANE_ELEMS_PER_BATCH,
        BATCH_CHUNK_ELEMS=HALF_PLANE_ELEMS_PER_BATCH,
        DST_INNER_BASE=HALF_PLANE_ELEMS_PER_BATCH,
        num_warps=num_warps,
    )

    return out


def replacement_func():
    return fused_cat_nearest_interpolate_stack_40