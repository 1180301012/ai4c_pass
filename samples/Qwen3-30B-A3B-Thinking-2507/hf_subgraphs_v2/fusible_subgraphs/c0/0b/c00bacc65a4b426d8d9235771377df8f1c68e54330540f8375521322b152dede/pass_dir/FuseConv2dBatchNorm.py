import torch
import triton
import triton.language as tl

# Pattern matching function
def pattern(in_5, in_4, in_0, in_1, in_3, in_2):
    conv2d = torch.conv2d(in_5, in_4, None, (1, 1), (1, 1), (1, 1), 1)
    tmp_6 = torch.nn.functional.batch_norm(conv2d, in_0, in_1, in_3, in_2, False, 0.1, 1e-05)
    return tmp_6

# Argument extraction function
def replacement_args(in_5, in_4, in_0, in_1, in_3, in_2):
    return (in_5, in_4, in_0, in_1, in_3, in_2)

# Triton kernel for fused convolution and batch normalization
@triton.jit
def fused_conv2d_batchnorm_kernel(
    X,        # Input tensor (batch, in_channels, H, W)
    W,        # Weights (out_channels, in_channels, 3, 3)
    mean,     # Running mean (out_channels)
    var,      # Running variance (out_channels)
    weight,   # Scale (out_channels)
    bias,     # Bias (out_channels)
    Y,        # Output tensor (batch, out_channels, H, W)
    batch_size,
    in_channels,
    out_channels,
    H,
    output_width,
    BLOCK_SIZE: tl.constexpr,
):
    # Thread IDs
    batch_id = tl.program_id(0)
    x = tl.program_id(1)  # Output row
    y = tl.program_id(2)  # Output col
    out_channel = tl.program_id(3)  # Output channel

    # Input position for convolution
    in_x_start = x - 1
    in_y_start = y - 1

    # Get mean, var, weight, bias for this channel
    m = tl.load(mean + out_channel)
    v = tl.load(var + out_channel)
    s = tl.load(weight + out_channel)
    b = tl.load(bias + out_channel)

    # Accumulator for convolution
    acc = tl.zeros((1,), dtype=tl.float32)

    # Perform convolution
    for c in range(in_channels):
        for kh in range(3):
            for kw in range(3):
                in_x = in_x_start + kh
                in_y = in_y_start + kw
                if in_x >= 0 and in_x < H and in_y >= 0 and in_y < W:
                    # Compute input tensor index
                    input_idx = batch_id * (in_channels * H * output_width) + c * (H * output_width) + in_x * output_width + in_y
                    x_val = tl.load(X + input_idx, dtype=tl.float16)
                    
                    # Compute weights tensor index
                    weight_idx = out_channel * (in_channels * 3 * 3) + c * (3 * 3) + kh * 3 + kw
                    w_val = tl.load(W + weight_idx, dtype=tl.float16)
                    
                    acc += x_val * w_val

    # Batch normalization
    normalized = (acc - m) / tl.sqrt(v + 1e-05)
    result = normalized * s + b

    # Store output
    output_idx = batch_id * (out_channels * H * output_width) + out_channel * (H * output_width) + x * output_width + y
    tl.store(Y + output_idx, result)

# Wrapper function
@torch.fx.wrap
def fused_conv2d_batchnorm(X, W, mean, var, weight, bias):
    batch_size, in_channels, H, W = X.shape
    out_channels = W.shape[0]
    
    # Create output tensor
    Y = torch.empty((batch_size, out_channels, H, W), dtype=X.dtype, device=X.device)
    
    # Set up kernel grid
    grid_x = batch_size
    grid_y = H
    grid_z = W
    grid_w = out_channels
    BLOCK_SIZE = 128

    # Launch kernel
    fused_conv2d_batchnorm_kernel[(grid_x, grid_y, grid_z, grid_w)](
        X, W, mean, var, weight, bias, Y,
        batch_size, in_channels, out_channels, H, W,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return Y

# Replacement function
def replacement_func():
    return fused_conv2d_batchnorm