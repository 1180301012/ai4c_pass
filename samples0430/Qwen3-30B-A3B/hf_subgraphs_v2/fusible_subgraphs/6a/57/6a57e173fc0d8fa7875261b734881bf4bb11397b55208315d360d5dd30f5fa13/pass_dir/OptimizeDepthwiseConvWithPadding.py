import torch
import triton
import triton.language as tl

# Pattern matching function
# Matches the depthwise convolution with specific parameters
# Note: Must match positional arguments exactly as in model.py
@torch.fx.wrap
def pattern(in_2, in_0):
    return torch.conv2d(in_2, in_0, None, (1, 1), (32, 0), (1, 1), 4)

# Argument extraction function
# Extracts all necessary shapes and parameters for the kernel
def replacement_args(in_2, in_0):
    B = in_2.shape[0]
    C = in_2.shape[1]
    H = in_2.shape[2]
    W = in_2.shape[3]
    K = in_0.shape[2]  # Kernel height (65)
    padding_left = 32
    return (in_2, in_0, B, C, H, W, K, padding_left)

# Triton kernel for depthwise convolution with padding
@triton.jit
def depthwise_conv2d_kernel(
    x_ptr,  # Input tensor [B, C, H, W]
    w_ptr,  # Weights tensor [C, K] (reshaped from [C, 1, K, 1])
    y_ptr,  # Output tensor [B, C, H, W]
    B: tl.int32,
    C: tl.int32,
    H: tl.int32,
    W: tl.int32,
    K: tl.int32,
    padding_left: tl.int32,
    BLOCK_H: tl.constexpr
):
    # Each block handles one (batch, channel, width) slice
    block_id = tl.program_id(0)
    b = block_id // (C * W)
    c = (block_id // W) % C
    w = block_id % W

    # Each thread handles one height position
    h = tl.thread_id(0)

    # Initialize accumulator to 0
    acc = tl.zeros((1,), dtype=tl.float32)

    # Process each kernel element (k from 0 to K-1)
    for k in tl.range(0, K):
        # Compute input index position: i = h + k - padding_left
        i = h + k - padding_left
        # Handle padding (zero if out of bounds)
        x_val = tl.load(x_ptr + b * C * H * W + c * H * W + i * W + w, 
                       mask=(i >= 0) & (i < H), other=0.0)
        # Load weight value
        w_val = tl.load(w_ptr + c * K + k)
        acc += x_val * w_val

    # Store result
    tl.store(y_ptr + b * C * H * W + c * H * W + h * W + w, acc)

# Kernel wrapper
@torch.fx.wrap
def kernel_wrapper(in_2, in_0, B, C, H, W, K, padding_left):
    # Allocate output tensor
    y = torch.empty_like(in_2)
    # Set kernel configuration
    grid = (B * C * W,)
    # Launch the Triton kernel
    depthwise_conv2d_kernel[grid](
        in_2,
        in_0.reshape(C, K),  # Reshape weight: [C, 1, K, 1] -> [C, K]
        y,
        B,
        C,
        H,
        W,
        K,
        padding_left,
        BLOCK_H=H  # Block size matches H for efficiency
    )
    return y

# Replacement function (returns the wrapper function)
def replacement_func():
    return kernel_wrapper