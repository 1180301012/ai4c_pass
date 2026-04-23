import torch
import triton
import triton.language as tl

def pattern(in_6, in_1, in_0):
    conv3d = torch.conv3d(in_6, in_1, in_0, (2, 16, 16), (0, 0, 0), (1, 1, 1), 1)
    tmp_7 = conv3d.flatten(2)
    tmp_8 = tmp_7.transpose(1, 2)
    return tmp_8

def replacement_args(in_6, in_1, in_0):
    return (in_6, in_1, in_0)

@triton.jit
def conv3d_kernel(
    in_ptr, weight_ptr, bias_ptr, out_ptr,
    batch, in_channels, in_d, in_h, in_w,
    out_channels, out_d, out_h, out_w,
    k_d, k_h, k_w,
    stride_d, stride_h, stride_w,
    padding_d, padding_h, padding_w,
    BLOCK_SIZE_OUT_C: tl.constexpr,
    BLOCK_SIZE_D: tl.constexpr,
    BLOCK_SIZE_H: tl.constexpr,
    BLOCK_SIZE_W: tl.constexpr
):
    out_d_idx = tl.program_id(0) * BLOCK_SIZE_D + tl.arange(0, BLOCK_SIZE_D)
    out_h_idx = tl.program_id(1) * BLOCK_SIZE_H + tl.arange(0, BLOCK_SIZE_H)
    out_w_idx = tl.program_id(2) * BLOCK_SIZE_W + tl.arange(0, BLOCK_SIZE_W)
    out_c_idx = tl.program_id(3) * BLOCK_SIZE_OUT_C + tl.arange(0, BLOCK_SIZE_OUT_C)

    out_d_mask = out_d_idx < out_d
    out_h_mask = out_h_idx < out_h
    out_w_mask = out_w_idx < out_w
    out_c_mask = out_c_idx < out_channels

    weight = tl.load(weight_ptr + 
                     (out_c_idx[:, None, None, None] * in_channels * k_d * k_h * k_w +
                      tl.arange(0, in_channels)[None, :, None, None] * k_d * k_h * k_w +
                      tl.arange(0, k_d)[None, None, :, None] * k_h * k_w +
                      tl.arange(0, k_h)[None, None, None, :] * k_w +
                      tl.arange(0, k_w)[None, None, None, :]),
                     mask=out_c_mask[:, None, None, None] & 
                     (tl.arange(0, in_channels)[None, :, None, None] < in_channels) &
                     (tl.arange(0, k_d)[None, None, :, None] < k_d) &
                     (tl.arange(0, k_h)[None, None, None, :] < k_h) &
                     (tl.arange(0, k_w)[None, None, None, :] < k_w))

    in_d_start = out_d_idx * stride_d - padding_d
    in_h_start = out_h_idx * stride_h - padding_h
    in_w_start = out_w_idx * stride_w - padding_w

    in_d_idx = in_d_start + tl.arange(0, k_d)
    in_h_idx = in_h_start + tl.arange(0, k_h)
    in_w_idx = in_w_start + tl.arange(0, k_w)

    in_d_mask = (in_d_idx >= 0) & (in_d_idx < in_d)
    in_h_mask = (in_h_idx >= 0) & (in_h_idx < in_h)
    in_w_mask = (in_w_idx >= 0) & (in_w_idx < in_w)

    in_data = tl.load(in_ptr + 
                      (in_channels * in_d * in_h * in_w * 0 +
                       tl.arange(0, in_channels)[:, None, None, None] * in_d * in_h * in_w +
                       in_d_idx[None, :, None, None] * in_h * in_w +
                       in_h_idx[None, None, :, None] * in_w +
                       in_w_idx[None, None, None, :]),
                      mask=in_d_mask & in_h_mask & in_w_mask)

    output = tl.sum(in_data * weight, axis=0)

    bias = tl.load(bias_ptr + out_c_idx, mask=out_c_mask)
    output += bias[None, None, None, None]

    out_offsets = (out_c_idx[:, None, None, None] * out_d * out_h * out_w +
                   out_d_idx[:, None, None, None] * out_h * out_w +
                   out_h_idx[None, :, None, None] * out_w +
                   out_w_idx[None, None, :, None])
    tl.store(out_ptr + out_offsets, output, mask=out_c_mask & out_d_mask & out_h_mask & out_w_mask)

@torch.fx.wrap
def triton_conv3d(in_6, in_1, in_0):
    batch, in_channels, in_d, in_h, in_w = in_6.shape
    out_channels = in_1.shape[0]
    k_d, k_h, k_w = in_1.shape[2], in_1.shape[3], in_1.shape[4]
    stride_d, stride_h, stride_w = 2, 16, 16
    padding_d, padding_h, padding_w = 0, 0, 0

    out_d = (in_d - k_d + 2 * padding_d) // stride_d + 1
    out_h = (in_h - k_h + 2 * padding_h) // stride_h + 1
    out_w = (in_w - k_w + 2 * padding_w) // stride_w + 1

    out = torch.empty((batch, out_channels, out_d, out_h, out_w), dtype=in_6.dtype, device=in_6.device)

    BLOCK_SIZE_OUT_C = 64
    BLOCK_SIZE_D = 8
    BLOCK_SIZE_H = 8
    BLOCK_SIZE_W = 8

    grid_d = (out_d + BLOCK_SIZE_D - 1) // BLOCK_SIZE_D
    grid_h = (out_h + BLOCK_SIZE_H - 1) // BLOCK_SIZE_H
    grid_w = (out_w + BLOCK_SIZE_W - 1) // BLOCK_SIZE_W
    grid_c = (out_channels + BLOCK_SIZE_OUT_C - 1) // BLOCK_SIZE_OUT_C

    conv3d_kernel[(grid_d, grid_h, grid_w, grid_c), 
                 (BLOCK_SIZE_D, BLOCK_SIZE_H, BLOCK_SIZE_W, BLOCK_SIZE_OUT_C)](
        in_6, in_1, in_0, out,
        batch, in_channels, in_d, in_h, in_w,
        out_channels, out_d, out_h, out_w,
        k_d, k_h, k_w,
        stride_d, stride_h, stride_w,
        padding_d, padding_h, padding_w,
        BLOCK_SIZE_OUT_C=BLOCK_SIZE_OUT_C,
        BLOCK_SIZE_D=BLOCK_SIZE_D,
        BLOCK_SIZE_H=BLOCK_SIZE_H,
        BLOCK_SIZE_W=BLOCK_SIZE_W
    )

    out = out.flatten(2)
    out = out.transpose(1, 2)
    return out

def replacement_func():
    return triton_conv3d