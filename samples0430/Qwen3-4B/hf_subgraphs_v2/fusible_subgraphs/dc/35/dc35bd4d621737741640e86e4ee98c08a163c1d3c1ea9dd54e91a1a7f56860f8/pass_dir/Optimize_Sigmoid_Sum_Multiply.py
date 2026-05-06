import torch
import triton
import triton.language as tl


def pattern(in_0, in_1):
    tmp_0 = in_1 * in_0
    tmp_1 = torch.sum(tmp_0, dim=1)
    tmp_2 = tmp_1.unsqueeze(1)
    tmp_3 = torch.sigmoid(tmp_2)
    return (tmp_3,)

def replacement_args(in_0, in_1):
    return (in_0, in_1)

@triton.jit
def optimized_kernel(
    in_0_ptr,
    in_1_ptr,
    out_ptr,
    B,
    C,
    H,
    W,
    BLOCK_SIZE: tl.constexpr,
):
    # Each thread handles a block of channels
    batch = tl.program_id(0)
    height = tl.program_id(1)
    width = tl.program_id(2)
    channel_start = 0
    channel_end = min(BLOCK_SIZE, C)
    total = 0.0
    for channel in range(channel_start, channel_end):
        # Load the values for this channel
        idx = batch * C * H * W + channel * H * W + height * W + width
        x = tl.load(in_0_ptr + idx)
        y = tl.load(in_1_ptr + idx)
        total += x * y
    out_val = 1.0 / (1.0 + tl.exp(-total))
    tl.store(out_ptr + (batch * H * W + height * W + width), out_val)

@torch.fx.wrap
def kernel_wrapper(in_0, in_1):
    B, C, H, W = in_0.shape
    out = torch.empty((B, 1, H, W), device=in_0.device, dtype=in_0.dtype)
    BLOCK_SIZE = 32
    num_blocks = (C + BLOCK_SIZE - 1) // BLOCK_SIZE
    optimized_kernel[(B, H, W, num_blocks)](
        in_0_ptr=in_0,
        in_1_ptr=in_1,
        out_ptr=out,
        B=B,
        C=C,
        H=H,
        W=W,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return out

def replacement_func():
    return kernel_wrapper