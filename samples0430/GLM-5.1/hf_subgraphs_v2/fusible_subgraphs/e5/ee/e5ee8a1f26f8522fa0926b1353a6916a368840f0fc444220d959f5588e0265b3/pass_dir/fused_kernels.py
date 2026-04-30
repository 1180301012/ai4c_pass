import torch
import triton
import triton.language as tl


def _get_triton_dtype(torch_dtype):
    """Convert torch dtype to Triton dtype."""
    if torch_dtype == torch.float32:
        return tl.float32
    elif torch_dtype == torch.float16:
        return tl.float16
    elif torch_dtype == torch.bfloat16:
        return tl.bfloat16
    else:
        return tl.float32


# ========== Depthwise Conv2D + GELU Kernel (3x3, padding=1) ==========

@triton.jit
def depthwise_conv2d_gelu_3x3_pad1_kernel(
    input_ptr, weight_ptr, bias_ptr, output_ptr,
    N, C, H, W,
    stride_in_n, stride_in_c, stride_in_h, stride_in_w,
    stride_wt_c, stride_wt_kh, stride_wt_kw,
    stride_out_n, stride_out_c, stride_out_h, stride_out_w,
    BLOCK_W: tl.constexpr,
    DTYPE: tl.constexpr,
):
    # 2D grid: (N*C*H, ceil(W/BLOCK_W))
    nch = tl.program_id(0)
    w_block = tl.program_id(1)

    # Decompose nch -> n, c, h
    nc = N * C
    n = nch // nc
    c = (nch % nc) // H
    h = (nch % nc) % H

    # Column range for this block
    w_start = w_block * BLOCK_W
    w_offsets = w_start + tl.arange(0, BLOCK_W)
    w_mask = w_offsets < W

    # Load bias (scalar, broadcast to BLOCK_W)
    b = tl.load(bias_ptr + c).to(tl.float32)

    # Initialize accumulator with bias
    acc = tl.full([BLOCK_W], b, dtype=tl.float32)

    # Conv2d: 3x3 kernel, padding=1
    for kh in range(3):
        ih = h + kh - 1  # Input height index with padding offset
        ih_valid = (ih >= 0) & (ih < H)  # Bounds check for padding

        for kw in range(3):
            iw = w_offsets + kw - 1  # Input width indices with padding offset
            iw_valid = (iw >= 0) & (iw < W)  # Bounds check for padding
            valid = ih_valid & iw_valid & w_mask

            # Load weight value for this (kh, kw) position
            wt_offset = c * stride_wt_c + kh * stride_wt_kh + kw * stride_wt_kw
            wt_val = tl.load(weight_ptr + wt_offset).to(tl.float32)

            # Load input value (0 if out of bounds due to padding)
            in_offset = n * stride_in_n + c * stride_in_c + ih * stride_in_h + iw * stride_in_w
            inp = tl.load(input_ptr + in_offset, mask=valid, other=0.0).to(tl.float32)

            # Accumulate
            acc = acc + wt_val * inp

    # GELU: x * 0.5 * (1 + erf(x / sqrt(2)))
    sqrt2 = 1.4142135623730951
    gelu_out = acc * 0.5 * (1.0 + tl.libdevice.erf(acc / sqrt2))

    # Cast to output dtype and store
    gelu_out = gelu_out.to(DTYPE)
    out_offset = n * stride_out_n + c * stride_out_c + h * stride_out_h + w_offsets * stride_out_w
    tl.store(output_ptr + out_offset, gelu_out, mask=w_mask)


@torch.fx.wrap
def depthwise_conv2d_gelu_3x3_pad1(bias, weight, input_tensor):
    """Fused depthwise conv2d (3x3, pad=1) + GELU. Dropout(0, False, False) is identity, so eliminated."""
    N, C, H, W = input_tensor.shape

    # Output shape same as input for depthwise conv with same-padding
    output = torch.empty_like(input_tensor)

    dtype = _get_triton_dtype(input_tensor.dtype)

    # Choose BLOCK_W based on W for good utilization
    if W <= 8:
        BLOCK_W = 8
    elif W <= 16:
        BLOCK_W = 16
    elif W <= 32:
        BLOCK_W = 32
    else:
        BLOCK_W = 64

    grid_nch = N * C * H
    grid_w = (W + BLOCK_W - 1) // BLOCK_W

    if grid_nch == 0 or grid_w == 0:
        return output

    s_in = input_tensor.stride()
    s_out = output.stride()
    s_wt = weight.stride()

    depthwise_conv2d_gelu_3x3_pad1_kernel[(grid_nch, grid_w)](
        input_ptr=input_tensor, weight_ptr=weight, bias_ptr=bias, output_ptr=output,
        N=N, C=C, H=H, W=W,
        stride_in_n=s_in[0], stride_in_c=s_in[1], stride_in_h=s_in[2], stride_in_w=s_in[3],
        stride_wt_c=s_wt[0], stride_wt_kh=s_wt[2], stride_wt_kw=s_wt[3],
        stride_out_n=s_out[0], stride_out_c=s_out[1], stride_out_h=s_out[2], stride_out_w=s_out[3],
        BLOCK_W=BLOCK_W,
        DTYPE=dtype,
    )

    return output


# ========== Pointwise Conv2D + GELU Kernel (1x1, no padding) ==========

@triton.jit
def pointwise_conv2d_gelu_1x1_kernel(
    input_ptr, weight_ptr, bias_ptr, output_ptr,
    N, C_IN: tl.constexpr, C_OUT: tl.constexpr, H, W,
    stride_in_n, stride_in_c, stride_in_h, stride_in_w,
    stride_wt_co, stride_wt_ci,
    stride_out_n, stride_out_co, stride_out_h, stride_out_w,
    BLOCK_NHW: tl.constexpr,
    BLOCK_CO: tl.constexpr,
    DTYPE: tl.constexpr,
):
    # 2D grid: (ceil(NHW/BLOCK_NHW), ceil(C_OUT/BLOCK_CO))
    nhw_block = tl.program_id(0)
    co_block = tl.program_id(1)

    NHW = N * H * W
    HW = H * W

    # Spatial position range
    nhw_start = nhw_block * BLOCK_NHW
    nhw_offsets = nhw_start + tl.arange(0, BLOCK_NHW)
    nhw_mask = nhw_offsets < NHW

    # Decompose nhw -> n, h, w
    n = nhw_offsets // HW
    hw = nhw_offsets % HW
    h = hw // W
    w = hw % W

    # Output channel range
    co_start = co_block * BLOCK_CO
    co_offsets = co_start + tl.arange(0, BLOCK_CO)
    co_mask = co_offsets < C_OUT

    # Load bias for this C_out block
    b = tl.load(bias_ptr + co_offsets, mask=co_mask, other=0.0).to(tl.float32)

    # Initialize accumulator with bias
    acc = tl.full([BLOCK_NHW, BLOCK_CO], 0.0, dtype=tl.float32) + b[None, :]

    # Compute dot product over C_IN (1x1 conv = matrix multiply per spatial position)
    for ci in range(C_IN):
        # Load input for all spatial positions at channel ci
        in_offset = n * stride_in_n + ci * stride_in_c + h * stride_in_h + w * stride_in_w
        inp = tl.load(input_ptr + in_offset, mask=nhw_mask, other=0.0).to(tl.float32)

        # Load weight for all C_out at channel ci
        wt_offset = co_offsets * stride_wt_co + ci * stride_wt_ci
        wt = tl.load(weight_ptr + wt_offset, mask=co_mask, other=0.0).to(tl.float32)

        # Accumulate: outer product
        acc = acc + inp[:, None] * wt[None, :]

    # GELU: x * 0.5 * (1 + erf(x / sqrt(2)))
    sqrt2 = 1.4142135623730951
    gelu_out = acc * 0.5 * (1.0 + tl.libdevice.erf(acc / sqrt2))

    # Cast to output dtype and store
    gelu_out = gelu_out.to(DTYPE)
    out_offset = n[:, None] * stride_out_n + co_offsets[None, :] * stride_out_co + h[:, None] * stride_out_h + w[:, None] * stride_out_w
    out_mask = nhw_mask[:, None] & co_mask[None, :]
    tl.store(output_ptr + out_offset, gelu_out, mask=out_mask)


@torch.fx.wrap
def pointwise_conv2d_gelu_1x1(bias, weight, input_tensor):
    """Fused pointwise conv2d (1x1, no padding) + GELU. Dropout(0, False, False) is identity, so eliminated."""
    N, C_IN, H, W = input_tensor.shape
    C_OUT = weight.shape[0]

    output = torch.empty((N, C_OUT, H, W), dtype=input_tensor.dtype, device=input_tensor.device)

    dtype = _get_triton_dtype(input_tensor.dtype)

    # Choose block sizes based on dimensions
    BLOCK_NHW = 32
    BLOCK_CO = 32

    if C_OUT <= 16:
        BLOCK_CO = 16
    elif C_OUT >= 128:
        BLOCK_CO = 64

    NHW = N * H * W
    grid_nhw = (NHW + BLOCK_NHW - 1) // BLOCK_NHW
    grid_co = (C_OUT + BLOCK_CO - 1) // BLOCK_CO

    if grid_nhw == 0 or grid_co == 0:
        return output

    s_in = input_tensor.stride()
    s_out = output.stride()
    s_wt = weight.stride()

    pointwise_conv2d_gelu_1x1_kernel[(grid_nhw, grid_co)](
        input_ptr=input_tensor, weight_ptr=weight, bias_ptr=bias, output_ptr=output,
        N=N, C_IN=C_IN, C_OUT=C_OUT, H=H, W=W,
        stride_in_n=s_in[0], stride_in_c=s_in[1], stride_in_h=s_in[2], stride_in_w=s_in[3],
        stride_wt_co=s_wt[0], stride_wt_ci=s_wt[1],
        stride_out_n=s_out[0], stride_out_co=s_out[1], stride_out_h=s_out[2], stride_out_w=s_out[3],
        BLOCK_NHW=BLOCK_NHW,
        BLOCK_CO=BLOCK_CO,
        DTYPE=dtype,
    )

    return output


# ========== Dispatch Wrapper (shared across all passes) ==========

@torch.fx.wrap
def fused_conv2d_gelu_drop_dispatch(*args):
    """Dispatch wrapper for routing between different conv2d + GELU variants.
    
    The last argument is a route string that determines which kernel to use.
    Route format:
    - "dw_*" for depthwise conv (3x3, pad=1) + GELU
    - "pw_1x1" for pointwise conv (1x1, no pad) + GELU
    """
    route = args[-1]
    tensor_args = args[:-1]  # (bias, weight, input)

    if route.startswith("dw_"):
        return depthwise_conv2d_gelu_3x3_pad1(*tensor_args)
    elif route == "pw_1x1":
        return pointwise_conv2d_gelu_1x1(*tensor_args)
    else:
        # Fallback: should not happen with valid routes
        raise ValueError(f"Unknown route: {route}")