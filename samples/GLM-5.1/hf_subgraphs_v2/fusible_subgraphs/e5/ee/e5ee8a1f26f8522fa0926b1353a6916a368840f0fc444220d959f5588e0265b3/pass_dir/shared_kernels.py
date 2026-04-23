import torch
import triton
import triton.language as tl


@triton.jit
def depthwise_conv2d_3x3_pad1_gelu_kernel(
    input_ptr, weight_ptr, bias_ptr, output_ptr,
    N, C, H, W,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """Depthwise conv2d with 3x3 kernel, padding=1, stride=1, dilation=1 + GELU fusion."""
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # Decompose flat output index into (n, c, h, w)
    chw = C * H * W
    hw = H * W
    n_idx = offsets // chw
    c_idx = (offsets // hw) % C
    h_idx = (offsets // W) % H
    w_idx = offsets % W

    # Load bias for this channel
    bias_val = tl.load(bias_ptr + c_idx, mask=mask, other=0.0).to(tl.float32)

    # Depthwise conv: each output channel c uses weight[c, 0, kh, kw] and input[n, c, ih, iw]
    # weight shape: [C, 1, 3, 3] -> flattened: c * 9 + kh * 3 + kw
    # input shape: [N, C, H, W] -> flattened: n * chw + c * hw + ih * W + iw
    acc = bias_val
    for kh in range(3):
        for kw in range(3):
            ih = h_idx + kh - 1  # padding = 1 means offset by -1
            iw = w_idx + kw - 1
            # Boundary check for input
            valid = (ih >= 0) & (ih < H) & (iw >= 0) & (iw < W)
            input_mask = mask & valid

            # Load weight value
            w_off = c_idx * 9 + kh * 3 + kw
            w_val = tl.load(weight_ptr + w_off, mask=mask, other=0.0).to(tl.float32)

            # Load input value
            i_off = n_idx * chw + c_idx * hw + ih * W + iw
            i_val = tl.load(input_ptr + i_off, mask=input_mask, other=0.0).to(tl.float32)

            acc = acc + w_val * i_val

    # Apply GELU: x * 0.5 * (1 + erf(x / sqrt(2)))
    # 1/sqrt(2) = 0.7071067811865476
    gelu_result = acc * 0.5 * (1.0 + tl.math.erf(acc * 0.7071067811865476))

    # Store output - convert back to input dtype (Triton handles dtype conversion at store)
    tl.store(output_ptr + offsets, gelu_result, mask=mask)


@triton.jit
def pointwise_conv2d_1x1_pad0_gelu_kernel(
    input_ptr, weight_ptr, bias_ptr, output_ptr,
    N, C_in, C_out, H, W,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """Pointwise conv2d with 1x1 kernel, padding=0, stride=1, dilation=1, groups=1 + GELU fusion."""
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # Output shape: [N, C_out, H, W]
    chw_out = C_out * H * W
    hw = H * W
    n_idx = offsets // chw_out
    c_out_idx = (offsets // hw) % C_out
    h_idx = (offsets // W) % H
    w_idx = offsets % W

    # Load bias for output channel
    bias_val = tl.load(bias_ptr + c_out_idx, mask=mask, other=0.0).to(tl.float32)

    # Compute dot product: sum over C_in of weight[c_out, c_in, 0, 0] * input[n, c_in, h, w]
    # weight shape: [C_out, C_in, 1, 1] -> flattened: c_out * C_in + c_in
    # input shape: [N, C_in, H, W] -> flattened: n * (C_in * H * W) + c_in * hw + h * W + w
    chw_in = C_in * H * W

    acc = bias_val
    for k_start in range(0, C_in, BLOCK_K):
        k_offsets = k_start + tl.arange(0, BLOCK_K)
        k_mask = k_offsets < C_in

        # Load weight values: weight[c_out, c_in]
        w_off = c_out_idx * C_in + k_offsets
        w_mask = mask & k_mask
        w_val = tl.load(weight_ptr + w_off, mask=w_mask, other=0.0).to(tl.float32)

        # Load input values: input[n, c_in, h, w]
        i_off = n_idx * chw_in + k_offsets * hw + h_idx * W + w_idx
        i_mask = mask & k_mask
        i_val = tl.load(input_ptr + i_off, mask=i_mask, other=0.0).to(tl.float32)

        acc = acc + w_val * i_val

    # Apply GELU
    gelu_result = acc * 0.5 * (1.0 + tl.math.erf(acc * 0.7071067811865476))

    tl.store(output_ptr + offsets, gelu_result, mask=mask)


@torch.fx.wrap
def depthwise_conv_gelu(bias, weight, input_tensor):
    """Fused depthwise conv2d (3x3, pad=1, stride=1) + GELU."""
    # input_tensor shape: [N, C, H, W]
    # weight shape: [C, 1, 3, 3] where C = groups
    # bias shape: [C]
    N, C, H, W = input_tensor.shape
    n_elements = N * C * H * W

    output = torch.empty_like(input_tensor)

    BLOCK_SIZE = 256
    num_programs = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE

    depthwise_conv2d_3x3_pad1_gelu_kernel[(num_programs,)](
        input_ptr=input_tensor,
        weight_ptr=weight,
        bias_ptr=bias,
        output_ptr=output,
        N=N, C=C, H=H, W=W,
        n_elements=n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    return output


@torch.fx.wrap
def pointwise_conv_gelu(bias, weight, input_tensor):
    """Fused pointwise conv2d (1x1, pad=0, stride=1, groups=1) + GELU."""
    # input_tensor shape: [N, C_in, H, W]
    # weight shape: [C_out, C_in, 1, 1]
    # bias shape: [C_out]
    N, C_in, H, W = input_tensor.shape
    C_out = weight.shape[0]
    n_elements = N * C_out * H * W

    output = torch.empty((N, C_out, H, W), dtype=input_tensor.dtype, device=input_tensor.device)

    BLOCK_SIZE = 256
    BLOCK_K = 32
    num_programs = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE

    pointwise_conv2d_1x1_pad0_gelu_kernel[(num_programs,)](
        input_ptr=input_tensor,
        weight_ptr=weight,
        bias_ptr=bias,
        output_ptr=output,
        N=N, C_in=C_in, C_out=C_out, H=H, W=W,
        n_elements=n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
        BLOCK_K=BLOCK_K,
    )

    return output


@torch.fx.wrap
def fused_conv_gelu_dispatch(*args):
    """Dispatch wrapper for fused conv2d + GELU kernels."""
    route = args[-1]
    tensor_args = args[:-1]
    # tensor_args = (bias, weight, input_tensor) based on replacement_args

    if route == "dw_g128":
        return depthwise_conv_gelu(*tensor_args)
    elif route == "dw_g256":
        return depthwise_conv_gelu(*tensor_args)
    elif route == "dw_g512":
        return depthwise_conv_gelu(*tensor_args)
    elif route == "dw_g1024":
        return depthwise_conv_gelu(*tensor_args)
    elif route == "dw_g2048":
        return depthwise_conv_gelu(*tensor_args)
    elif route == "pw_g1":
        return pointwise_conv_gelu(*tensor_args)
    else:
        raise ValueError(f"Unknown route: {route}")