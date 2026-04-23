import torch
import triton
import triton.language as tl


@triton.jit
def fused_depthwise_conv2d_mean_kernel(
    input_ptr, weight_ptr, conv_out_ptr, mean_out_ptr,
    N, C, H_in, W_in, H_out, W_out,
    stride_h: tl.constexpr, stride_w: tl.constexpr,
    BLOCK_OW: tl.constexpr,
):
    n = tl.program_id(0)
    c = tl.program_id(1)

    # Load 9 weights for this channel, promote to float32 for accumulation
    w_ptrs = weight_ptr + c * 9 + tl.arange(0, 9)
    weights = tl.load(w_ptrs).to(tl.float32)

    # Compute base offsets for this (n, c) pair
    in_base = n * C * H_in * W_in + c * H_in * W_in
    out_base = n * C * H_out * W_out + c * H_out * W_out

    # float32 scalar accumulator for mean
    sum_acc = 0.0

    oh = 0
    while oh < H_out:
        ow_start = 0
        while ow_start < W_out:
            ow = ow_start + tl.arange(0, BLOCK_OW)
            mask_ow = ow < W_out

            # Accumulate conv result in float32
            result = tl.zeros([BLOCK_OW], dtype=tl.float32)

            for kh in range(3):
                for kw in range(3):
                    ih = oh * stride_h + kh - 1
                    iw = ow * stride_w + kw - 1

                    # Check if input location is valid (handles zero-padding)
                    valid = (ih >= 0) & (ih < H_in) & (iw >= 0) & (iw < W_in) & mask_ow

                    in_offset = in_base + ih * W_in + iw
                    val = tl.load(input_ptr + in_offset, mask=valid, other=0.0).to(tl.float32)
                    result += val * weights[kh * 3 + kw]

            # Store conv output (Triton casts float32 to pointer dtype automatically)
            out_offset = out_base + oh * W_out + ow
            tl.store(conv_out_ptr + out_offset, result, mask=mask_ow)

            # Accumulate sum for mean computation
            sum_acc = sum_acc + tl.sum(result * mask_ow.to(tl.float32))
            ow_start += BLOCK_OW
        oh += 1

    # Compute and store mean value
    mean_val = sum_acc / (H_out * W_out)
    mean_offset = n * C + c  # mean shape [N, C, 1, 1], stored contiguously
    tl.store(mean_out_ptr + mean_offset, mean_val)


@torch.fx.wrap
def _fused_conv2d_mean_s2_g384(input_tensor, weight_tensor):
    N, C_in, H_in, W_in = input_tensor.shape
    groups = weight_tensor.shape[0]
    C_out = groups
    padding = 1
    kernel_size = 3
    H_out = (H_in + 2 * padding - kernel_size) // 2 + 1
    W_out = (W_in + 2 * padding - kernel_size) // 2 + 1
    conv_out = torch.empty((N, C_out, H_out, W_out), dtype=input_tensor.dtype, device=input_tensor.device)
    mean_out = torch.empty((N, C_out, 1, 1), dtype=input_tensor.dtype, device=input_tensor.device)
    BLOCK_OW = 32
    grid = (N, C_out)
    fused_depthwise_conv2d_mean_kernel[grid](
        input_tensor, weight_tensor, conv_out, mean_out,
        N, C_out, H_in, W_in, H_out, W_out,
        stride_h=2, stride_w=2,
        BLOCK_OW=BLOCK_OW,
    )
    return conv_out, mean_out


@torch.fx.wrap
def _fused_conv2d_mean_s1_g256(input_tensor, weight_tensor):
    N, C_in, H_in, W_in = input_tensor.shape
    groups = weight_tensor.shape[0]
    C_out = groups
    padding = 1
    kernel_size = 3
    H_out = (H_in + 2 * padding - kernel_size) // 1 + 1
    W_out = (W_in + 2 * padding - kernel_size) // 1 + 1
    conv_out = torch.empty((N, C_out, H_out, W_out), dtype=input_tensor.dtype, device=input_tensor.device)
    mean_out = torch.empty((N, C_out, 1, 1), dtype=input_tensor.dtype, device=input_tensor.device)
    BLOCK_OW = 32
    grid = (N, C_out)
    fused_depthwise_conv2d_mean_kernel[grid](
        input_tensor, weight_tensor, conv_out, mean_out,
        N, C_out, H_in, W_in, H_out, W_out,
        stride_h=1, stride_w=1,
        BLOCK_OW=BLOCK_OW,
    )
    return conv_out, mean_out


@torch.fx.wrap
def _fused_conv2d_mean_s2_g256(input_tensor, weight_tensor):
    N, C_in, H_in, W_in = input_tensor.shape
    groups = weight_tensor.shape[0]
    C_out = groups
    padding = 1
    kernel_size = 3
    H_out = (H_in + 2 * padding - kernel_size) // 2 + 1
    W_out = (W_in + 2 * padding - kernel_size) // 2 + 1
    conv_out = torch.empty((N, C_out, H_out, W_out), dtype=input_tensor.dtype, device=input_tensor.device)
    mean_out = torch.empty((N, C_out, 1, 1), dtype=input_tensor.dtype, device=input_tensor.device)
    BLOCK_OW = 32
    grid = (N, C_out)
    fused_depthwise_conv2d_mean_kernel[grid](
        input_tensor, weight_tensor, conv_out, mean_out,
        N, C_out, H_in, W_in, H_out, W_out,
        stride_h=2, stride_w=2,
        BLOCK_OW=BLOCK_OW,
    )
    return conv_out, mean_out


@torch.fx.wrap
def _fused_conv2d_mean_s1_g384(input_tensor, weight_tensor):
    N, C_in, H_in, W_in = input_tensor.shape
    groups = weight_tensor.shape[0]
    C_out = groups
    padding = 1
    kernel_size = 3
    H_out = (H_in + 2 * padding - kernel_size) // 1 + 1
    W_out = (W_in + 2 * padding - kernel_size) // 1 + 1
    conv_out = torch.empty((N, C_out, H_out, W_out), dtype=input_tensor.dtype, device=input_tensor.device)
    mean_out = torch.empty((N, C_out, 1, 1), dtype=input_tensor.dtype, device=input_tensor.device)
    BLOCK_OW = 32
    grid = (N, C_out)
    fused_depthwise_conv2d_mean_kernel[grid](
        input_tensor, weight_tensor, conv_out, mean_out,
        N, C_out, H_in, W_in, H_out, W_out,
        stride_h=1, stride_w=1,
        BLOCK_OW=BLOCK_OW,
    )
    return conv_out, mean_out


@torch.fx.wrap
def _fused_conv2d_mean_s1_g768(input_tensor, weight_tensor):
    N, C_in, H_in, W_in = input_tensor.shape
    groups = weight_tensor.shape[0]
    C_out = groups
    padding = 1
    kernel_size = 3
    H_out = (H_in + 2 * padding - kernel_size) // 1 + 1
    W_out = (W_in + 2 * padding - kernel_size) // 1 + 1
    conv_out = torch.empty((N, C_out, H_out, W_out), dtype=input_tensor.dtype, device=input_tensor.device)
    mean_out = torch.empty((N, C_out, 1, 1), dtype=input_tensor.dtype, device=input_tensor.device)
    BLOCK_OW = 32
    grid = (N, C_out)
    fused_depthwise_conv2d_mean_kernel[grid](
        input_tensor, weight_tensor, conv_out, mean_out,
        N, C_out, H_in, W_in, H_out, W_out,
        stride_h=1, stride_w=1,
        BLOCK_OW=BLOCK_OW,
    )
    return conv_out, mean_out


# Dispatch wrapper shared across all passes
@torch.fx.wrap
def dispatch_fused_conv2d_mean(input_tensor, weight_tensor, route):
    if route == "s1_g256":
        return _fused_conv2d_mean_s1_g256(input_tensor, weight_tensor)
    elif route == "s2_g256":
        return _fused_conv2d_mean_s2_g256(input_tensor, weight_tensor)
    elif route == "s1_g384":
        return _fused_conv2d_mean_s1_g384(input_tensor, weight_tensor)
    elif route == "s2_g384":
        return _fused_conv2d_mean_s2_g384(input_tensor, weight_tensor)
    elif route == "s1_g768":
        return _fused_conv2d_mean_s1_g768(input_tensor, weight_tensor)
    else:
        raise ValueError(f"Unknown route: {route}")


def pattern(in_0, in_1):
    conv2d = torch.conv2d(in_1, in_0, None, (2, 2), (1, 1), (1, 1), 256)
    tmp_2 = conv2d.mean((2, 3), keepdim=True)
    return conv2d, tmp_2


def replacement_args(in_0, in_1):
    return (in_1, in_0, "s2_g256")


def replacement_func():
    return dispatch_fused_conv2d_mean