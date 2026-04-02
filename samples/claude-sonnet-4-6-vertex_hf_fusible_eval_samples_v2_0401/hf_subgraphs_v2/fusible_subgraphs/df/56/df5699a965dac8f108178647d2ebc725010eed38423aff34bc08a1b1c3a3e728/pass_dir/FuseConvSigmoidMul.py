import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_HW': 128}, num_warps=4),
        triton.Config({'BLOCK_HW': 256}, num_warps=4),
        triton.Config({'BLOCK_HW': 512}, num_warps=8),
        triton.Config({'BLOCK_HW': 1024}, num_warps=8),
    ],
    key=['HW'],
)
@triton.jit
def fused_conv_sigmoid_scale_kernel(
    x_ptr, w_ptr, b_ptr, y_ptr, out_ptr,
    N, C_OUT, HW,
    C_IN: tl.constexpr,
    C_IN_ROUNDED: tl.constexpr,
    BLOCK_HW: tl.constexpr,
):
    nc_idx = tl.program_id(0)
    hw_block_idx = tl.program_id(1)

    n = nc_idx // C_OUT
    c_out = nc_idx % C_OUT

    # Load weight row for this c_out [C_IN elements]
    c_in_range = tl.arange(0, C_IN_ROUNDED)
    c_in_mask = c_in_range < C_IN
    w_vals = tl.load(w_ptr + c_out * C_IN + c_in_range, mask=c_in_mask, other=0.0).to(tl.float32)
    x_vals = tl.load(x_ptr + n * C_IN + c_in_range, mask=c_in_mask, other=0.0).to(tl.float32)

    # Dot product
    acc = tl.sum(x_vals * w_vals, axis=0)

    # Add bias
    acc += tl.load(b_ptr + c_out).to(tl.float32)

    # Sigmoid
    scale = tl.sigmoid(acc)

    # Scale spatial positions
    hw_start = hw_block_idx * BLOCK_HW
    hw_offsets = hw_start + tl.arange(0, BLOCK_HW)
    mask = hw_offsets < HW

    base = (n * C_OUT + c_out) * HW
    y_vals = tl.load(y_ptr + base + hw_offsets, mask=mask, other=0.0)
    dtype = y_vals.dtype
    out_vals = y_vals.to(tl.float32) * scale
    tl.store(out_ptr + base + hw_offsets, out_vals.to(dtype), mask=mask)


@torch.fx.wrap
def fused_conv_sigmoid_scale(x, weight, bias, y):
    N = x.shape[0]
    C_IN = weight.shape[1]   # 10
    C_OUT = weight.shape[0]  # 40
    H = y.shape[2]
    W = y.shape[3]
    HW = H * W

    # Flatten spatial dims for Triton kernel
    x_flat = x.reshape(N, C_IN)
    w_flat = weight.reshape(C_OUT, C_IN)
    y_flat = y.reshape(N, C_OUT, HW)
    out = torch.empty_like(y_flat)

    # Round C_IN up to next power of 2 for tl.arange
    C_IN_ROUNDED = 1
    while C_IN_ROUNDED < C_IN:
        C_IN_ROUNDED *= 2

    grid = lambda meta: (N * C_OUT, triton.cdiv(HW, meta['BLOCK_HW']))
    fused_conv_sigmoid_scale_kernel[grid](
        x_flat, w_flat, bias, y_flat, out,
        N, C_OUT, HW,
        C_IN=C_IN,
        C_IN_ROUNDED=C_IN_ROUNDED,
    )

    return out.reshape(N, C_OUT, H, W)


def pattern(in_6, in_1, in_0, in_5):
    conv2d = torch.conv2d(in_6, in_1, in_0, (1, 1), (0, 0), (1, 1), 1)
    tmp_3 = torch.sigmoid(conv2d)
    tmp_4 = in_5 * tmp_3
    return tmp_4


def replacement_args(in_6, in_1, in_0, in_5):
    return (in_6, in_1, in_0, in_5)


def replacement_func():
    return fused_conv_sigmoid_scale