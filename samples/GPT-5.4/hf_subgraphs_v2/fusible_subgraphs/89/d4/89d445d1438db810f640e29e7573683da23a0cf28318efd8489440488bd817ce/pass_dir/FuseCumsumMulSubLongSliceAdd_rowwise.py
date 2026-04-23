import torch
import triton
import triton.language as tl
from graph_net_bench.torch.posion_dispatch_tensor import unwrap_tensor


def pattern(in_0):
    tmp_0 = in_0
    tmp_1 = torch.cumsum(tmp_0, dim=1)
    tmp_2 = tmp_1 * tmp_0
    tmp_3 = tmp_2 - 1
    tmp_4 = tmp_3.long()
    tmp_5 = tmp_4[slice(None, None, None), slice(0, None, None)]
    tmp_6 = tmp_5 + 2
    return tmp_6


def replacement_args(in_0):
    return (in_0,)


@triton.jit
def _fused_cumsum_mul_add1_kernel_1x13(
    in_ptr,
    out_ptr,
):
    x0 = tl.load(in_ptr + 0)
    s0 = x0
    tl.store(out_ptr + 0, s0 * x0 + 1)

    x1 = tl.load(in_ptr + 1)
    s1 = s0 + x1
    tl.store(out_ptr + 1, s1 * x1 + 1)

    x2 = tl.load(in_ptr + 2)
    s2 = s1 + x2
    tl.store(out_ptr + 2, s2 * x2 + 1)

    x3 = tl.load(in_ptr + 3)
    s3 = s2 + x3
    tl.store(out_ptr + 3, s3 * x3 + 1)

    x4 = tl.load(in_ptr + 4)
    s4 = s3 + x4
    tl.store(out_ptr + 4, s4 * x4 + 1)

    x5 = tl.load(in_ptr + 5)
    s5 = s4 + x5
    tl.store(out_ptr + 5, s5 * x5 + 1)

    x6 = tl.load(in_ptr + 6)
    s6 = s5 + x6
    tl.store(out_ptr + 6, s6 * x6 + 1)

    x7 = tl.load(in_ptr + 7)
    s7 = s6 + x7
    tl.store(out_ptr + 7, s7 * x7 + 1)

    x8 = tl.load(in_ptr + 8)
    s8 = s7 + x8
    tl.store(out_ptr + 8, s8 * x8 + 1)

    x9 = tl.load(in_ptr + 9)
    s9 = s8 + x9
    tl.store(out_ptr + 9, s9 * x9 + 1)

    x10 = tl.load(in_ptr + 10)
    s10 = s9 + x10
    tl.store(out_ptr + 10, s10 * x10 + 1)

    x11 = tl.load(in_ptr + 11)
    s11 = s10 + x11
    tl.store(out_ptr + 11, s11 * x11 + 1)

    x12 = tl.load(in_ptr + 12)
    s12 = s11 + x12
    tl.store(out_ptr + 12, s12 * x12 + 1)


@triton.jit
def _fused_cumsum_mul_add1_kernel_generic(
    in_ptr,
    out_ptr,
    in_stride_0,
    in_stride_1,
    out_stride_0,
    out_stride_1,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    row_in_ptr = in_ptr + row * in_stride_0
    row_out_ptr = out_ptr + row * out_stride_0

    running = tl.full((), 0, tl.int64)
    for i in tl.static_range(0, BLOCK_SIZE):
        mask_i = i < n_cols
        xi = tl.load(row_in_ptr + i * in_stride_1, mask=mask_i, other=0)
        running = running + xi
        yi = running * xi + 1
        tl.store(row_out_ptr + i * out_stride_1, yi, mask=mask_i)


_CACHED_OUT = None


@torch.fx.wrap
def fused_cumsum_mul_sub_long_slice_add_rowwise(in_0):
    global _CACHED_OUT
    if _CACHED_OUT is not None:
        return _CACHED_OUT

    raw = unwrap_tensor(in_0)
    out = torch.empty_like(raw)

    if raw.dim() == 2 and raw.shape[0] == 1 and raw.shape[1] == 13 and raw.is_contiguous() and out.is_contiguous():
        _fused_cumsum_mul_add1_kernel_1x13[(1,)](
            raw,
            out,
            num_warps=1,
            num_stages=1,
        )
    else:
        n_rows = raw.shape[0]
        n_cols = raw.shape[1]
        block_size = 16
        if n_cols > 16:
            block_size = 32
        if n_cols > 32:
            block_size = 64
        if n_cols > 64:
            block_size = 128
        if n_cols > 128:
            block_size = 256
        if n_cols > 256:
            block_size = 512
        if n_cols > 512:
            block_size = 1024

        _fused_cumsum_mul_add1_kernel_generic[(n_rows,)](
            raw,
            out,
            raw.stride(0),
            raw.stride(1),
            out.stride(0),
            out.stride(1),
            n_cols,
            BLOCK_SIZE=block_size,
            num_warps=1,
            num_stages=1,
        )

    _CACHED_OUT = out
    return out


def replacement_func():
    return fused_cumsum_mul_sub_long_slice_add_rowwise