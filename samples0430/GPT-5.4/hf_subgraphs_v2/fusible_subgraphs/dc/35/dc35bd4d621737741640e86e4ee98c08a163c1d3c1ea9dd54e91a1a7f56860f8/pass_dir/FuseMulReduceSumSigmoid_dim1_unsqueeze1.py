import torch
import triton
import triton.language as tl


C_FIXED = 64
HW_FIXED = 64 * 64


def pattern(in_0, in_1):
    tmp_0 = in_1 * in_0
    tmp_1 = torch.sum(tmp_0, dim=1)
    tmp_2 = tmp_1.unsqueeze(1)
    tmp_3 = torch.sigmoid(tmp_2)
    return tmp_3


def replacement_args(in_0, in_1):
    return (in_0, in_1)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_HW': 64}, num_warps=2, num_stages=2),
        triton.Config({'BLOCK_HW': 128}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_HW': 256}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_HW': 512}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_HW': 1024}, num_warps=8, num_stages=2),
    ],
    key=['HW'],
)
@triton.jit
def _fused_mul_reduce_sigmoid_contig64_kernel(
    x_ptr,
    y_ptr,
    out_ptr,
    HW,
    stride_x_n,
    stride_y_n,
    stride_out_n,
    BLOCK_HW: tl.constexpr,
):
    pid_hw = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_hw = pid_hw * BLOCK_HW + tl.arange(0, BLOCK_HW)
    hw_mask = offs_hw < HW

    x_batch_ptr = x_ptr + pid_n * stride_x_n + offs_hw
    y_batch_ptr = y_ptr + pid_n * stride_y_n + offs_hw

    acc = tl.zeros([BLOCK_HW], dtype=tl.float32)

    for c0 in tl.static_range(0, 64, 8):
        x0 = tl.load(x_batch_ptr + (c0 + 0) * HW, mask=hw_mask, other=0.0)
        y0 = tl.load(y_batch_ptr + (c0 + 0) * HW, mask=hw_mask, other=0.0)
        x1 = tl.load(x_batch_ptr + (c0 + 1) * HW, mask=hw_mask, other=0.0)
        y1 = tl.load(y_batch_ptr + (c0 + 1) * HW, mask=hw_mask, other=0.0)
        x2 = tl.load(x_batch_ptr + (c0 + 2) * HW, mask=hw_mask, other=0.0)
        y2 = tl.load(y_batch_ptr + (c0 + 2) * HW, mask=hw_mask, other=0.0)
        x3 = tl.load(x_batch_ptr + (c0 + 3) * HW, mask=hw_mask, other=0.0)
        y3 = tl.load(y_batch_ptr + (c0 + 3) * HW, mask=hw_mask, other=0.0)
        x4 = tl.load(x_batch_ptr + (c0 + 4) * HW, mask=hw_mask, other=0.0)
        y4 = tl.load(y_batch_ptr + (c0 + 4) * HW, mask=hw_mask, other=0.0)
        x5 = tl.load(x_batch_ptr + (c0 + 5) * HW, mask=hw_mask, other=0.0)
        y5 = tl.load(y_batch_ptr + (c0 + 5) * HW, mask=hw_mask, other=0.0)
        x6 = tl.load(x_batch_ptr + (c0 + 6) * HW, mask=hw_mask, other=0.0)
        y6 = tl.load(y_batch_ptr + (c0 + 6) * HW, mask=hw_mask, other=0.0)
        x7 = tl.load(x_batch_ptr + (c0 + 7) * HW, mask=hw_mask, other=0.0)
        y7 = tl.load(y_batch_ptr + (c0 + 7) * HW, mask=hw_mask, other=0.0)

        acc += (
            x0.to(tl.float32) * y0.to(tl.float32)
            + x1.to(tl.float32) * y1.to(tl.float32)
            + x2.to(tl.float32) * y2.to(tl.float32)
            + x3.to(tl.float32) * y3.to(tl.float32)
            + x4.to(tl.float32) * y4.to(tl.float32)
            + x5.to(tl.float32) * y5.to(tl.float32)
            + x6.to(tl.float32) * y6.to(tl.float32)
            + x7.to(tl.float32) * y7.to(tl.float32)
        )

    out = tl.sigmoid(acc)
    out_ptrs = out_ptr + pid_n * stride_out_n + offs_hw
    tl.store(out_ptrs, out, mask=hw_mask)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_HW': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_HW': 128}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_HW': 256}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_HW': 256}, num_warps=8, num_stages=2),
    ],
    key=['HW', 'CHANNELS'],
)
@triton.jit
def _fused_mul_reduce_sigmoid_generic_kernel(
    x_ptr,
    y_ptr,
    out_ptr,
    W,
    HW,
    stride_x_n,
    stride_x_c,
    stride_x_h,
    stride_x_w,
    stride_y_n,
    stride_y_c,
    stride_y_h,
    stride_y_w,
    stride_out_n,
    CHANNELS: tl.constexpr,
    BLOCK_HW: tl.constexpr,
):
    pid_hw = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_hw = pid_hw * BLOCK_HW + tl.arange(0, BLOCK_HW)
    mask = offs_hw < HW
    offs_h = offs_hw // W
    offs_w = offs_hw % W

    base_x_n = pid_n * stride_x_n
    base_y_n = pid_n * stride_y_n

    acc = tl.zeros([BLOCK_HW], dtype=tl.float32)
    for c in tl.static_range(0, CHANNELS):
        x_idx = base_x_n + c * stride_x_c + offs_h * stride_x_h + offs_w * stride_x_w
        y_idx = base_y_n + c * stride_y_c + offs_h * stride_y_h + offs_w * stride_y_w
        x = tl.load(x_ptr + x_idx, mask=mask, other=0.0)
        y = tl.load(y_ptr + y_idx, mask=mask, other=0.0)
        acc += x.to(tl.float32) * y.to(tl.float32)

    out = tl.sigmoid(acc)
    out_idx = pid_n * stride_out_n + offs_hw
    tl.store(out_ptr + out_idx, out, mask=mask)


@torch.fx.wrap
def fused_mul_reduce_sigmoid(in_0, in_1):
    n = in_0.shape[0]
    c = in_0.shape[1]
    h = in_0.shape[2]
    w = in_0.shape[3]
    hw = h * w

    out = torch.empty((n, 1, h, w), device=in_0.device, dtype=in_0.dtype)

    if (
        c == C_FIXED
        and h == 64
        and w == 64
        and in_0.stride(3) == 1
        and in_0.stride(2) == w
        and in_0.stride(1) == hw
        and in_1.stride(3) == 1
        and in_1.stride(2) == w
        and in_1.stride(1) == hw
        and out.stride(3) == 1
        and out.stride(2) == w
        and out.stride(1) == hw
    ):
        grid = lambda meta: (triton.cdiv(hw, meta['BLOCK_HW']), n)
        _fused_mul_reduce_sigmoid_contig64_kernel[grid](
            in_0,
            in_1,
            out,
            hw,
            in_0.stride(0),
            in_1.stride(0),
            out.stride(0),
        )
    else:
        grid = lambda meta: (triton.cdiv(hw, meta['BLOCK_HW']), n)
        _fused_mul_reduce_sigmoid_generic_kernel[grid](
            in_0,
            in_1,
            out,
            w,
            hw,
            in_0.stride(0),
            in_0.stride(1),
            in_0.stride(2),
            in_0.stride(3),
            in_1.stride(0),
            in_1.stride(1),
            in_1.stride(2),
            in_1.stride(3),
            out.stride(0),
            CHANNELS=c,
        )
    return out


def replacement_func():
    return fused_mul_reduce_sigmoid