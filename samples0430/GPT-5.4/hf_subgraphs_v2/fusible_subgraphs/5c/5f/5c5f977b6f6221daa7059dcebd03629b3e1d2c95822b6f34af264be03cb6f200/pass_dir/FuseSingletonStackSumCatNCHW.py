import torch
import triton
import triton.language as tl


# Match the exact post-conv pattern present in the target graphs.
def pattern(conv_out, other):
    tmp_3 = torch.stack([conv_out], dim=0)
    tmp_4 = tmp_3.sum(dim=0)
    tmp_5 = torch.cat([tmp_4, other], 1)
    return tmp_5


# We only need the conv result and the tensor concatenated after it.
def replacement_args(conv_out, other):
    return (conv_out, other)


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": 256}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_SIZE": 512}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_SIZE": 1024}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_SIZE": 2048}, num_warps=8, num_stages=2),
    ],
    key=["TOTAL_ELEMS"],
)
@triton.jit
def _copy_into_cat_contig_kernel(
    src_ptr,
    out_ptr,
    TOTAL_ELEMS,
    IN_BATCH_ELEMS,
    OUT_BATCH_ELEMS,
    OUT_BASE_OFFSET,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < TOTAL_ELEMS

    batch_idx = offsets // IN_BATCH_ELEMS
    rem = offsets - batch_idx * IN_BATCH_ELEMS
    out_offsets = batch_idx * OUT_BATCH_ELEMS + OUT_BASE_OFFSET + rem

    vals = tl.load(src_ptr + offsets, mask=mask)
    tl.store(out_ptr + out_offsets, vals, mask=mask)


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_HW": 256}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_HW": 512}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_HW": 512}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_HW": 1024}, num_warps=8, num_stages=2),
    ],
    key=["HW"],
)
@triton.jit
def _copy_into_cat_nchw_kernel(
    src_ptr,
    out_ptr,
    C,
    W,
    HW,
    src_sn,
    src_sc,
    src_sh,
    src_sw,
    out_sn,
    out_sc,
    out_sh,
    out_sw,
    c_offset,
    BLOCK_HW: tl.constexpr,
):
    pid_hw = tl.program_id(0)
    pid_nc = tl.program_id(1)

    hw_offsets = pid_hw * BLOCK_HW + tl.arange(0, BLOCK_HW)
    mask = hw_offsets < HW

    h = hw_offsets // W
    w = hw_offsets - h * W

    n = pid_nc // C
    c = pid_nc - n * C

    src_offsets = n * src_sn + c * src_sc + h * src_sh + w * src_sw
    out_offsets = n * out_sn + (c + c_offset) * out_sc + h * out_sh + w * out_sw

    vals = tl.load(src_ptr + src_offsets, mask=mask)
    tl.store(out_ptr + out_offsets, vals, mask=mask)


@torch.fx.wrap
def fused_singleton_stack_sum_cat_nchw(conv_out, other):
    n = conv_out.shape[0]
    c1 = conv_out.shape[1]
    h = conv_out.shape[2]
    w = conv_out.shape[3]
    c2 = other.shape[1]

    out = torch.empty((n, c1 + c2, h, w), device=conv_out.device, dtype=conv_out.dtype)
    hw = h * w
    out_batch_elems = (c1 + c2) * hw

    if conv_out.is_contiguous() and other.is_contiguous() and out.is_contiguous():
        if conv_out.numel() != 0:
            conv_total = conv_out.numel()
            conv_batch = c1 * hw
            grid_1 = lambda meta: (triton.cdiv(conv_total, meta["BLOCK_SIZE"]),)
            _copy_into_cat_contig_kernel[grid_1](
                conv_out,
                out,
                conv_total,
                conv_batch,
                out_batch_elems,
                0,
            )

        if other.numel() != 0:
            other_total = other.numel()
            other_batch = c2 * hw
            grid_2 = lambda meta: (triton.cdiv(other_total, meta["BLOCK_SIZE"]),)
            _copy_into_cat_contig_kernel[grid_2](
                other,
                out,
                other_total,
                other_batch,
                out_batch_elems,
                c1 * hw,
            )
        return out

    if conv_out.numel() != 0:
        grid_1 = lambda meta: (triton.cdiv(hw, meta["BLOCK_HW"]), n * c1)
        _copy_into_cat_nchw_kernel[grid_1](
            conv_out,
            out,
            c1,
            w,
            hw,
            conv_out.stride(0),
            conv_out.stride(1),
            conv_out.stride(2),
            conv_out.stride(3),
            out.stride(0),
            out.stride(1),
            out.stride(2),
            out.stride(3),
            0,
        )

    if other.numel() != 0:
        grid_2 = lambda meta: (triton.cdiv(hw, meta["BLOCK_HW"]), n * c2)
        _copy_into_cat_nchw_kernel[grid_2](
            other,
            out,
            c2,
            w,
            hw,
            other.stride(0),
            other.stride(1),
            other.stride(2),
            other.stride(3),
            out.stride(0),
            out.stride(1),
            out.stride(2),
            out.stride(3),
            c1,
        )

    return out


def replacement_func():
    return fused_singleton_stack_sum_cat_nchw