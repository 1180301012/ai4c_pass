import torch
import triton
import triton.language as tl

# Pattern matching function

def pattern(in_6, in_1, in_0, in_5):
    conv2d = torch.conv2d(in_6, in_1, in_0, (1, 1), (0, 0), (1, 1), 1)
    tmp_3 = torch.sigmoid(conv2d)
    tmp_4 = in_5 * tmp_3
    return tmp_4

# Argument extraction function
def replacement_args(in_6, in_1, in_0, in_5):
    return (in_6, in_1, in_0, in_5)

# Triton kernel implementation
@triton.jit
def fused_conv_sigmoid_mul_kernel(
    in_6_ptr, in_1_ptr, in_0_ptr, in_5_ptr, out_ptr,
    batch, channels, height, width, in_channels,
    BLOCK_SIZE: tl.constexpr,
):
    # 1. Compute convolution for (b, c)
    bid = tl.program_id(0)  # batch * channels
    pid = tl.program_id(1)  # spatial
    b = bid // channels
    c = bid % channels

    acc = tl.zeros((), dtype=tl.float32)
    for k in range(0, in_channels, BLOCK_SIZE):
        k_end = tl.minimum(k + BLOCK_SIZE, in_channels)
        in_6_val = tl.load(in_6_ptr + b * in_channels + k, mask=k < in_channels, other=0.0)
        in_1_val = tl.load(in_1_ptr + c * in_channels + k, mask=k < in_channels, other=0.0)
        acc += in_6_val * in_1_val
    acc += tl.load(in_0_ptr + c)
    sigmoid_out = 1.0 / (1.0 + tl.exp(-acc))

    # 2. Broadcast to spatial dimensions
    start_idx = pid * BLOCK_SIZE
    for i in range(BLOCK_SIZE):
        idx_in_spatial = start_idx + i
        if idx_in_spatial < height * width:
            h = idx_in_spatial // width
            w = idx_in_spatial % width
            out_idx = b * channels * height * width + c * height * width + h * width + w
            in_5_val = tl.load(in_5_ptr + out_idx)
            out_val = sigmoid_out * in_5_val
            tl.store(out_ptr + out_idx, out_val)

# Kernel wrapper
@torch.fx.wrap
def fused_conv_sigmoid_mul(in_6, in_1, in_0, in_5):
    batch = in_6.shape[0]
    channels = in_1.shape[0]  # out_channels
    in_channels = in_1.shape[1]  # in_channels
    height = in_5.shape[2]
    width = in_5.shape[3]

    out = torch.empty_like(in_5)

    num_blocks_x = batch * channels
    num_blocks_y = (height * width + 127) // 128

    fused_conv_sigmoid_mul_kernel[(num_blocks_x, num_blocks_y)](
        in_6, in_1, in_0, in_5, out,
        batch, channels, height, width, in_channels,
        BLOCK_SIZE=128
    )

    return out

# Replacement function (NO arguments, returns function reference)
def replacement_func():
    return fused_conv_sigmoid_mul