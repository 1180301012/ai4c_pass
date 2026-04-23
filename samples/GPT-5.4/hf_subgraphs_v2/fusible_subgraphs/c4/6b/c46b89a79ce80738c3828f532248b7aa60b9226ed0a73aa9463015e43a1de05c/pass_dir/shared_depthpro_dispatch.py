import torch
import triton
import triton.language as tl


@triton.jit
def _flat_cat_cast_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    out_ptr,
    size_a,
    size_b,
    size_c,
    total_size,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < total_size

    ab_size = size_a + size_b

    mask_a = mask & (offsets < size_a)
    mask_b = mask & (offsets >= size_a) & (offsets < ab_size)
    mask_c = mask & (offsets >= ab_size)

    vals = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    vals = tl.where(mask_a, tl.load(a_ptr + offsets, mask=mask_a, other=0).to(tl.float32), vals)
    vals = tl.where(mask_b, tl.load(b_ptr + (offsets - size_a), mask=mask_b, other=0).to(tl.float32), vals)
    vals = tl.where(mask_c, tl.load(c_ptr + (offsets - ab_size), mask=mask_c, other=0).to(tl.float32), vals)

    tl.store(out_ptr + offsets, vals.to(tl.float16), mask=mask)


@triton.jit
# src is arbitrary-strided 4D [N, C, H, W], dst is contiguous fp16 [N_total, C, H, W]
def _copy_strided_nchw_to_contig_fp16_kernel(
    src_ptr,
    dst_ptr,
    dst_n_offset,
    C,
    H,
    W,
    src_s0,
    src_s1,
    src_s2,
    src_s3,
    dst_CHW,
    dst_HW,
    total_rows,
    BLOCK_X: tl.constexpr,
):
    pid_x = tl.program_id(0)
    row = tl.program_id(1)

    x = pid_x * BLOCK_X + tl.arange(0, BLOCK_X)
    mask = x < W

    chw = C * H
    n = row // chw
    rem = row % chw
    c = rem // H
    h = rem % H

    src_idx = n * src_s0 + c * src_s1 + h * src_s2 + x * src_s3
    dst_idx = (dst_n_offset + n) * dst_CHW + c * dst_HW + h * W + x

    vals = tl.load(src_ptr + src_idx, mask=mask, other=0)
    tl.store(dst_ptr + dst_idx, vals.to(tl.float16), mask=mask)


@triton.jit
def _copy_patch_rows_kernel(
    src_ptr,
    out_ptr,
    dst_patch_offset,
    patch_cols,
    stride_hw,
    src_hw,
    total_rows,
    BLOCK_X: tl.constexpr,
):
    pid_x = tl.program_id(0)
    row_id = tl.program_id(1)

    x_offsets = pid_x * BLOCK_X + tl.arange(0, BLOCK_X)
    mask = x_offsets < 384

    C = 3
    KH = 384
    KW = 384
    PATCH_ROWS = C * KH
    PATCH_ELEMS = C * KH * KW

    patch_id = row_id // PATCH_ROWS
    row_in_patch = row_id % PATCH_ROWS
    c = row_in_patch // KH
    ky = row_in_patch % KH

    patch_y = patch_id // patch_cols
    patch_x = patch_id % patch_cols

    src_base = c * src_hw * src_hw + (patch_y * stride_hw + ky) * src_hw + patch_x * stride_hw
    dst_base = (dst_patch_offset + patch_id) * PATCH_ELEMS + c * KH * KW + ky * KW

    vals = tl.load(src_ptr + src_base + x_offsets, mask=mask, other=0)
    tl.store(out_ptr + dst_base + x_offsets, vals.to(tl.float16), mask=mask)


@torch.fx.wrap
def depthpro_dispatch(*args):
    route = args[-1]

    if route == "cat_cast":
        a, b, c, _ = args
        na = a.shape[0]
        nb = b.shape[0]
        nc = c.shape[0]
        C = a.shape[1]
        H = a.shape[2]
        W = a.shape[3]

        out = torch.empty((na + nb + nc, C, H, W), device=a.device, dtype=torch.float16)

        if (
            a.stride(3) == 1 and b.stride(3) == 1 and c.stride(3) == 1 and
            a.stride(2) == W and b.stride(2) == W and c.stride(2) == W and
            a.stride(1) == H * W and b.stride(1) == H * W and c.stride(1) == H * W and
            a.stride(0) == C * H * W and b.stride(0) == C * H * W and c.stride(0) == C * H * W
        ):
            size_a = na * C * H * W
            size_b = nb * C * H * W
            size_c = nc * C * H * W
            total_size = size_a + size_b + size_c
            BLOCK_SIZE = 4096
            grid = (triton.cdiv(total_size, BLOCK_SIZE),)
            _flat_cat_cast_kernel[grid](
                a,
                b,
                c,
                out,
                size_a,
                size_b,
                size_c,
                total_size,
                BLOCK_SIZE=BLOCK_SIZE,
                num_warps=8,
                num_stages=4,
            )
            return out

        BLOCK_X = 256
        dst_CHW = C * H * W
        dst_HW = H * W

        total_rows_a = na * C * H
        total_rows_b = nb * C * H
        total_rows_c = nc * C * H

        grid_a = (triton.cdiv(W, BLOCK_X), total_rows_a)
        grid_b = (triton.cdiv(W, BLOCK_X), total_rows_b)
        grid_c = (triton.cdiv(W, BLOCK_X), total_rows_c)

        _copy_strided_nchw_to_contig_fp16_kernel[grid_a](
            a,
            out,
            0,
            C,
            H,
            W,
            a.stride(0),
            a.stride(1),
            a.stride(2),
            a.stride(3),
            dst_CHW,
            dst_HW,
            total_rows_a,
            BLOCK_X=BLOCK_X,
            num_warps=4,
            num_stages=2,
        )
        _copy_strided_nchw_to_contig_fp16_kernel[grid_b](
            b,
            out,
            na,
            C,
            H,
            W,
            b.stride(0),
            b.stride(1),
            b.stride(2),
            b.stride(3),
            dst_CHW,
            dst_HW,
            total_rows_b,
            BLOCK_X=BLOCK_X,
            num_warps=4,
            num_stages=2,
        )
        _copy_strided_nchw_to_contig_fp16_kernel[grid_c](
            c,
            out,
            na + nb,
            C,
            H,
            W,
            c.stride(0),
            c.stride(1),
            c.stride(2),
            c.stride(3),
            dst_CHW,
            dst_HW,
            total_rows_c,
            BLOCK_X=BLOCK_X,
            num_warps=4,
            num_stages=2,
        )
        return out

    if route == "full_pack":
        in_0, in_1, in_2, _ = args
        out = torch.empty((35, 3, 384, 384), device=in_0.device, dtype=torch.float16)

        BLOCK_X = 256
        grid_in2 = (triton.cdiv(384, BLOCK_X), 25 * 3 * 384)
        grid_in1 = (triton.cdiv(384, BLOCK_X), 9 * 3 * 384)
        grid_in0 = (triton.cdiv(384, BLOCK_X), 1 * 3 * 384)

        _copy_patch_rows_kernel[grid_in2](
            in_2,
            out,
            0,
            5,
            288,
            1536,
            25 * 3 * 384,
            BLOCK_X=BLOCK_X,
            num_warps=4,
            num_stages=2,
        )
        _copy_patch_rows_kernel[grid_in1](
            in_1,
            out,
            25,
            3,
            192,
            768,
            9 * 3 * 384,
            BLOCK_X=BLOCK_X,
            num_warps=4,
            num_stages=2,
        )
        _copy_patch_rows_kernel[grid_in0](
            in_0,
            out,
            34,
            1,
            0,
            384,
            1 * 3 * 384,
            BLOCK_X=BLOCK_X,
            num_warps=4,
            num_stages=2,
        )
        return out

    return args[0]