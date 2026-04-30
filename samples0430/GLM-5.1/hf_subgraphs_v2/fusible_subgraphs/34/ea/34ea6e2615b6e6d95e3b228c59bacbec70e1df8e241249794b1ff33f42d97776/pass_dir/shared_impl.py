import torch
import triton
import triton.language as tl


@triton.jit
def dw_conv_add_kernel(
    input_ptr, weight_ptr, bias_ptr, output_ptr,
    C, H, W,
    stride_ic, stride_ih, stride_iw,
    stride_wc, stride_wkh, stride_wkw,
    BLOCK_C: tl.constexpr,
):
    hw_idx = tl.program_id(0)
    c_block = tl.program_id(1)
    c_start = c_block * BLOCK_C

    h = hw_idx // W
    w = hw_idx % W

    c_offsets = c_start + tl.arange(0, BLOCK_C)
    mask_c = c_offsets < C

    # Accumulate conv result in float32 for numerical stability
    result = tl.zeros([BLOCK_C], dtype=tl.float32)

    # 3x3 depthwise convolution with padding=1
    for kh in range(3):
        ih = h + kh - 1
        for kw in range(3):
            iw = w + kw - 1

            # Check boundary for zero padding
            valid = (ih >= 0) and (ih < H) and (iw >= 0) and (iw < W)

            if valid:
                input_offsets = c_offsets * stride_ic + ih * stride_ih + iw * stride_iw
                input_vals = tl.load(input_ptr + input_offsets, mask=mask_c, other=0.0).to(tl.float32)
            else:
                input_vals = tl.zeros([BLOCK_C], dtype=tl.float32)

            weight_offsets = c_offsets * stride_wc + kh * stride_wkh + kw * stride_wkw
            weight_vals = tl.load(weight_ptr + weight_offsets, mask=mask_c, other=0.0).to(tl.float32)

            result += input_vals * weight_vals

    # Add conv bias
    bias_vals = tl.load(bias_ptr + c_offsets, mask=mask_c, other=0.0).to(tl.float32)
    result += bias_vals

    # Add residual (input at same spatial position)
    residual_offsets = c_offsets * stride_ic + h * stride_ih + w * stride_iw
    residual_vals = tl.load(input_ptr + residual_offsets, mask=mask_c, other=0.0).to(tl.float32)
    result += residual_vals

    # Store output in [1, HW, C] contiguous format: offset = hw_idx * C + c
    output_offsets = hw_idx * C + c_offsets
    tl.store(output_ptr + output_offsets, result, mask=mask_c)


@triton.jit
def layer_norm_kernel(
    input_ptr, gamma_ptr, beta_ptr, output_ptr,
    C,
    stride_i_hw, stride_i_c,
    eps: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    hw_idx = tl.program_id(0)
    row_ptr = input_ptr + hw_idx * stride_i_hw

    # First pass: compute mean
    _mean = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for off in range(0, C, BLOCK_SIZE):
        c_offsets = off + tl.arange(0, BLOCK_SIZE)
        mask = c_offsets < C
        x = tl.load(row_ptr + c_offsets * stride_i_c, mask=mask, other=0.0).to(tl.float32)
        _mean += tl.where(mask, x, 0.0)
    mean = tl.sum(_mean, axis=0) / C

    # Second pass: compute variance
    _var = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for off in range(0, C, BLOCK_SIZE):
        c_offsets = off + tl.arange(0, BLOCK_SIZE)
        mask = c_offsets < C
        x = tl.load(row_ptr + c_offsets * stride_i_c, mask=mask, other=0.0).to(tl.float32)
        x_centered = x - mean
        _var += tl.where(mask, x_centered * x_centered, 0.0)
    var = tl.sum(_var, axis=0) / C
    rstd = 1.0 / tl.sqrt(var + eps)

    # Third pass: normalize, scale, shift, and store
    for off in range(0, C, BLOCK_SIZE):
        c_offsets = off + tl.arange(0, BLOCK_SIZE)
        mask = c_offsets < C
        x = tl.load(row_ptr + c_offsets * stride_i_c, mask=mask, other=0.0).to(tl.float32)
        g = tl.load(gamma_ptr + c_offsets, mask=mask, other=0.0).to(tl.float32)
        b = tl.load(beta_ptr + c_offsets, mask=mask, other=0.0).to(tl.float32)
        y = (x - mean) * rstd * g + b
        # Store in [HW, 1, C] contiguous format: offset = hw_idx * C + c
        tl.store(output_ptr + hw_idx * C + c_offsets, y, mask=mask)


@torch.fx.wrap
def fused_impl(in_0, in_1, in_2, in_3, in_4, route):
    if route == "route_768":
        C_val = 768
    elif route == "route_1024":
        C_val = 1024
    else:
        raise ValueError(f"Unknown route: {route}")

    H = in_4.shape[2]
    W = in_4.shape[3]
    HW = H * W
    dtype = in_4.dtype
    device = in_4.device

    # Allocate output tensors
    # tmp_7: [1, HW, C] - pre-layer-norm result (conv + add + reshape)
    out_7 = torch.empty((1, HW, C_val), dtype=dtype, device=device)
    # tmp_9/tmp_10: [HW, 1, C] - post-layer-norm result (transposed)
    out_9 = torch.empty((HW, 1, C_val), dtype=dtype, device=device)

    # Launch conv + add + reshape kernel
    BLOCK_C = 64
    num_c_blocks = triton.cdiv(C_val, BLOCK_C)
    grid_conv = (HW, num_c_blocks)

    dw_conv_add_kernel[grid_conv](
        input_ptr=in_4,
        weight_ptr=in_3,
        bias_ptr=in_2,
        output_ptr=out_7,
        C=C_val,
        H=H,
        W=W,
        stride_ic=in_4.stride()[1],
        stride_ih=in_4.stride()[2],
        stride_iw=in_4.stride()[3],
        stride_wc=in_3.stride()[0],
        stride_wkh=in_3.stride()[2],
        stride_wkw=in_3.stride()[3],
        BLOCK_C=BLOCK_C,
    )

    # Launch layer norm + transpose kernel
    BLOCK_SIZE = 256
    grid_ln = (HW,)

    layer_norm_kernel[grid_ln](
        input_ptr=out_7,
        gamma_ptr=in_1,
        beta_ptr=in_0,
        output_ptr=out_9,
        C=C_val,
        stride_i_hw=out_7.stride()[1],
        stride_i_c=out_7.stride()[2],
        eps=1e-5,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    # tmp_10 and tmp_9 are identical (both are transpose(0,1) of the same tensor)
    out_10 = out_9

    return (out_7, out_10, out_9)


def replacement_func():
    return fused_impl