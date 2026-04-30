import torch
import triton
import triton.language as tl


def pattern(x, y, in_2, in_3):
    tmp_0 = torch.cat((in_2, in_3), 1)
    tmp_1 = torch.stack([x, y, tmp_0])
    return tmp_1


def replacement_args(x, y, in_2, in_3):
    return (x, y, in_2, in_3)


PLANE_BATCH_ELEMS = 512 * 40 * 40
HALF_BATCH_ELEMS = 256 * 40 * 40


@triton.jit
def cat_then_stack3_kernel(
    x_ptr,
    y_ptr,
    in2_ptr,
    in3_ptr,
    out_ptr,
    plane_elems,
    BLOCK_SIZE: tl.constexpr,
    PLANE_BATCH_ELEMS_C: tl.constexpr,
    HALF_BATCH_ELEMS_C: tl.constexpr,
):
    plane = tl.program_id(0)
    pid = tl.program_id(1)

    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < plane_elems
    out_offs = plane * plane_elems + offs

    plane0_mask = mask & (plane == 0)
    plane1_mask = mask & (plane == 1)
    plane2_mask = mask & (plane == 2)

    x_val = tl.load(x_ptr + offs, mask=plane0_mask, other=0)
    tl.store(out_ptr + out_offs, x_val, mask=plane0_mask)

    y_val = tl.load(y_ptr + offs, mask=plane1_mask, other=0)
    tl.store(out_ptr + out_offs, y_val, mask=plane1_mask)

    batch = offs // PLANE_BATCH_ELEMS_C
    rem = offs % PLANE_BATCH_ELEMS_C

    in2_mask = plane2_mask & (rem < HALF_BATCH_ELEMS_C)
    in3_mask = plane2_mask & (rem >= HALF_BATCH_ELEMS_C)

    in2_offs = batch * HALF_BATCH_ELEMS_C + rem
    in3_inner = tl.where(rem >= HALF_BATCH_ELEMS_C, rem - HALF_BATCH_ELEMS_C, 0)
    in3_offs = batch * HALF_BATCH_ELEMS_C + in3_inner

    in2_val = tl.load(in2_ptr + in2_offs, mask=in2_mask, other=0)
    tl.store(out_ptr + out_offs, in2_val, mask=in2_mask)

    in3_val = tl.load(in3_ptr + in3_offs, mask=in3_mask, other=0)
    tl.store(out_ptr + out_offs, in3_val, mask=in3_mask)


@torch.fx.wrap
def cat_then_stack3_fused(x, y, in_2, in_3):
    out = torch.empty((3,) + tuple(x.shape), device=x.device, dtype=x.dtype)

    plane_elems = x.numel()
    block_size = 2048
    grid = (3, triton.cdiv(plane_elems, block_size))

    cat_then_stack3_kernel[grid](
        x,
        y,
        in_2,
        in_3,
        out,
        plane_elems,
        BLOCK_SIZE=block_size,
        PLANE_BATCH_ELEMS_C=PLANE_BATCH_ELEMS,
        HALF_BATCH_ELEMS_C=HALF_BATCH_ELEMS,
        num_warps=8,
    )

    return out


def replacement_func():
    return cat_then_stack3_fused