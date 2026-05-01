import torch
import triton
import triton.language as tl

def pattern(in_0, in_1, in_2, in_3, in_4, in_5, in_6):
    conv = torch.conv2d(in_6, in_4, None, (1, 1), (1, 1), (1, 1), 1)
    bn = torch.nn.functional.batch_norm(conv, in_0, in_1, in_3, in_2, False, 0.1, 1e-05)
    relu = torch.nn.functional.leaky_relu(bn, 0.01, True)
    out = relu + in_5
    return out

def replacement_args(in_0, in_1, in_2, in_3, in_4, in_5, in_6):
    return (in_0, in_1, in_2, in_3, in_4, in_5, in_6)

@triton.jit
def fused_conv_bn_relu_add_kernel(
    input_ptr,
    weights_ptr,
    running_mean_ptr,
    running_var_ptr,
    bn_weight_ptr,
    bn_bias_ptr,
    residual_ptr,
    output_ptr,
    batch_size,
    in_channels,
    out_channels,
    H,
    W,
    kernel_h,
    kernel_w,
    eps,
    alpha,
    BLOCK_H: tl.constexpr,
    BLOCK_W: tl.constexpr,
    BLOCK_OC: tl.constexpr
):
    # Coordinates for output block
    oc = tl.program_id(0)
    h = tl.program_id(1)
    w = tl.program_id(2)

    # Compute convolution for current output channel and position
    acc = tl.zeros((BLOCK_H, BLOCK_W), dtype=tl.float32)
    for ic in range(in_channels):
        for kh in range(kernel_h):
            for kw in range(kernel_w):
                # Load input data
                input_val = tl.load(
                    input_ptr + (oc * batch_size * in_channels * H * W) + 
                    (ic * H * W) + (h + kh) * W + (w + kw),
                    mask=(h + kh < H) & (w + kw < W),
                    other=0.0
                )
                # Load weight
                weight_val = tl.load(
                    weights_ptr + (oc * in_channels * kernel_h * kernel_w) + 
                    (ic * kernel_h * kernel_w) + kh * kernel_w + kw
                )
                acc += input_val * weight_val

    # Apply BatchNorm
    running_mean = tl.load(running_mean_ptr + oc)
    running_var = tl.load(running_var_ptr + oc)
    scale = 1.0 / tl.sqrt(running_var + eps)
    bn_weight = tl.load(bn_weight_ptr + oc)
    bn_bias = tl.load(bn_bias_ptr + oc)
    
    # Apply BatchNorm and LeakyReLU
    x_norm = (acc * scale) * bn_weight + bn_bias
    relu_out = tl.where(x_norm > 0, x_norm, alpha * x_norm)

    # Add residual connection
    residual = tl.load(residual_ptr + (oc * batch_size * H * W) + (h * W + w))
    output_val = relu_out + residual

    # Store output
    tl.store(
        output_ptr + (oc * batch_size * H * W) + (h * W + w),
        output_val,
        mask=(h < H) & (w < W)
    )

@torch.fx.wrap
def fused_conv_bn_relu_add_wrapper(*args):
    in_0, in_1, in_2, in_3, in_4, in_5, in_6 = args
    batch_size = in_6.shape[0]
    in_channels = in_6.shape[1]
    H = in_6.shape[2]
    W = in_6.shape[3]
    out_channels = in_3.shape[0]

    # Allocate output tensor
    output = torch.empty(
        (batch_size, out_channels, H, W),
        dtype=in_6.dtype,
        device=in_6.device
    )

    # Configure kernel grid
    grid = (out_channels, H, W)

    # Launch kernel
    fused_conv_bn_relu_add_kernel[grid](
        in_6,
        in_4,
        in_0,
        in_1,
        in_3,
        in_2,
        in_5,
        output,
        batch_size,
        in_channels,
        out_channels,
        H,
        W,
        3, 3,  # kernel size
        1e-05,  # eps
        0.01,   # alpha
        BLOCK_H=32,
        BLOCK_W=32,
        BLOCK_OC=8
    )

    return output

def replacement_func():
    return fused_conv_bn_relu_add_wrapper