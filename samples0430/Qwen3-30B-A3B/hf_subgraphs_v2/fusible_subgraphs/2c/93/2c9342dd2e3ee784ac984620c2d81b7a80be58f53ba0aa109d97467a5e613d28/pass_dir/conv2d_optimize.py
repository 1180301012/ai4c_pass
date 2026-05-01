import torch
import triton
import triton.language as tl


def pattern(in_0, in_1, in_2):
    conv2d = torch.conv2d(in_2, in_1, in_0, (1, 1), (0, 0), (1, 1), 1)
    return conv2d


def replacement_args(in_0, in_1, in_2):
    return (in_0, in_1, in_2)


@triton.jit
def conv2d_kernel(
    in_2_ptr, in_1_ptr, in_0_ptr,
    out_ptr,
    batch, H, W,
    in_2_stride_batch, in_2_stride_channel, in_2_stride_h, in_2_stride_w,
    in_1_stride,  # weight is 1D
    in_0_stride,
    out_stride_batch, out_stride_h, out_stride_w,
    BLOCK_SIZE: tl.constexpr = 64
):
    # Each thread handles one channel
    c = tl.program_id(0)
    if c >= 64:
        return

    # Calculate spatial position
    pid = tl.program_id(0)
    batch_idx = pid // (H * W)
    pos = pid % (H * W)
    y = pos // W
    x = pos % W

    # Compute input address for (batch, y, x, c)
    input_ptr = in_2_ptr + batch_idx * in_2_stride_batch + c * in_2_stride_channel + y * in_2_stride_h + x * in_2_stride_w
    weight_ptr = in_1_ptr + c * in_1_stride
    bias = tl.load(in_0_ptr, mask=True)

    # Load input value and weight
    in_val = tl.load(input_ptr, mask=c < 64)
    w_val = tl.load(weight_ptr, mask=c < 64)
    val = in_val * w_val

    # Reduce across channels (64 channels)
    total = val
    total += bias

    # Store output
    out_ptr = out_ptr + batch_idx * out_stride_batch + y * out_stride_h + x * out_stride_w
    tl.store(out_ptr, total)


@torch.fx.wrap
def conv2d_optimized(in_0, in_1, in_2):
    # Input tensor shapes
    batch, in_channels, H, W = in_2.shape
    # Weight: [1, 64, 1, 1] → reshape to [64]
    # Initialize output tensor
    conv2d_out = torch.empty(batch, 1, H, W, dtype=in_2.dtype, device=in_2.device)

    # Configure kernel launch
    grid = (batch * H * W,)
    BLOCK_SIZE = 64

    # Launch kernel
    conv2d_kernel[grid](
        in_2_ptr=in_2,
        in_1_ptr=in_1,
        in_0_ptr=in_0,
        out_ptr=conv2d_out,
        batch=batch, H=H, W=W,
        in_2_stride_batch=in_2.stride(0), in_2_stride_channel=in_2.stride(1), in_2_stride_h=in_2.stride(2), in_2_stride_w=in_2.stride(3),
        in_1_stride=in_1.stride(1),
        in_0_stride=in_0.stride(0),
        out_stride_batch=conv2d_out.stride(0), out_stride_h=conv2d_out.stride(2), out_stride_w=conv2d_out.stride(3),
        BLOCK_SIZE=BLOCK_SIZE,
    )

    return conv2d_out

def replacement_func():
    return conv2d_optimized