import torch
import triton
import triton.language as tl

# Pattern matching function
# Matches conv2d followed by batch_norm and relu
def pattern(x, w, bias, running_mean, running_var, weight_bn, bias_bn):
    conv = torch.conv2d(x, w, bias, (1, 1), (1, 1), (1, 1), 1)
    bn = torch.nn.functional.batch_norm(conv, running_mean, running_var, weight_bn, bias_bn, False, 0.1, 1e-05)
    relu = torch.nn.functional.relu(bn, inplace=False)
    return relu

# Argument extraction function
# Extracts all required arguments from matched nodes
def replacement_args(x, w, bias, running_mean, running_var, weight_bn, bias_bn):
    return (x, w, bias, running_mean, running_var, weight_bn, bias_bn)

# Triton kernel for fused conv2d + batch norm + relu
@triton.jit
def fused_conv_bn_relu_kernel(
    x_ptr,
    w_ptr,
    bias_ptr,
    out_ptr,
    H,
    W,
    input_channels,
    output_channels,
    kernel_size,
    BLOCK_H,
    BLOCK_W,
    BLOCK_C,
):
    # Calculate output coordinates
    x = tl.program_id(0) * BLOCK_W + tl.arange(0, BLOCK_W)
    y = tl.program_id(1) * BLOCK_H + tl.arange(0, BLOCK_H)
    c = tl.program_id(2) * BLOCK_C + tl.arange(0, BLOCK_C)

    # Mask for valid output coordinates
    mask_x = (x < W) & (x >= 0)
    mask_y = (y < H) & (y >= 0)
    mask = mask_y[:, None] & mask_x[None, :]

    # Load bias values
    bias = tl.load(bias_ptr + c, mask=(c < output_channels), other=0.0)

    # Initialize accumulator
    acc = tl.zeros((BLOCK_H, BLOCK_W, BLOCK_C), dtype=tl.float32)

    # Process input channels and kernel positions
    for d in range(input_channels):
        for k in range(kernel_size):
            for l in range(kernel_size):
                # Calculate input coordinates (with padding)
                inp_x = x + l - 1
                inp_y = y + k - 1
                
                # Check input bounds
                mask_x_in = (inp_x >= 0) & (inp_x < W)
                mask_y_in = (inp_y >= 0) & (inp_y < H)
                mask_in = mask_x_in & mask_y_in

                # Load input value
                inp_val = tl.load(
                    x_ptr + (0 * H * W + inp_y * W + inp_x) * input_channels + d,
                    mask=mask_in,
                    other=0.0
                )

                # Load weight value
                w_val = tl.load(
                    w_ptr + (c * input_channels * kernel_size * kernel_size + d * kernel_size * kernel_size + k * kernel_size + l),
                    mask=(c < output_channels),
                    other=0.0
                )

                # Accumulate
                acc += inp_val * w_val

    # Apply bias and ReLU
    acc += bias
    acc = tl.maximum(acc, 0)

    # Store output
    tl.store(
        out_ptr + (0 * H * W + y * W + x) * output_channels + c,
        acc,
        mask=mask & (c < output_channels)
    )

# Kernel wrapper
@torch.fx.wrap
def fused_conv_bn_relu(x, w, bias, running_mean, running_var, weight_bn, bias_bn):
    # Precompute adjusted weights and bias (batch norm fused into conv)
    eps = 1e-05
    new_weight = weight_bn / torch.sqrt(running_var + eps)
    new_bias = bias_bn - running_mean * new_weight

    # Scale weights by new_weight
    w_new = w * new_weight.view(-1, 1, 1, 1)

    # Allocate output tensor
    batch_size, _, H, W = x.shape
    out = torch.empty((batch_size, w_new.shape[0], H, W), dtype=x.dtype)

    # Configure kernel launch parameters
    BLOCK_H, BLOCK_W, BLOCK_C = 8, 8, 32
    grid_h = (H + BLOCK_H - 1) // BLOCK_H
    grid_w = (W + BLOCK_W - 1) // BLOCK_W
    grid_c = (w_new.shape[0] + BLOCK_C - 1) // BLOCK_C

    # Launch kernel
    fused_conv_bn_relu_kernel[(grid_h, grid_w, grid_c)](
        x, w_new, new_bias, out, H, W, w_new.shape[1], w_new.shape[0], 3, BLOCK_H, BLOCK_W, BLOCK_C
    )
    return out

# Replacement function returns the kernel wrapper
def replacement_func():
    return fused_conv_bn_relu