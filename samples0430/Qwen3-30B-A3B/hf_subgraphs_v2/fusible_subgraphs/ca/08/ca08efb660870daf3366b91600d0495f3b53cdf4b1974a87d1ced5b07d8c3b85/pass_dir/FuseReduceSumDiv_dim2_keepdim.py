import torch
import triton
import triton.language as tl

# Sum kernel
@triton.jit
def sum_kernel(
    in_ptr,
    sum_ptr,
    batch: tl.int32,
    channels: tl.int32,
    height: tl.int32,
    width: tl.int32,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    b = pid // (channels * width)
    c = (pid // width) % channels
    w = pid % width

    total = 0.0
    for h in range(height):
        idx = b * channels * height * width + c * height * width + h * width + w
        x = tl.load(in_ptr + idx)
        total += x

    sum_idx = b * channels * 1 * width + c * 1 * width + 0 * width + w
    tl.store(sum_ptr + sum_idx, total)

# Division kernel
@triton.jit
def div_kernel(
    in_ptr,
    sum_ptr,
    out_ptr,
    batch: tl.int32,
    channels: tl.int32,
    height: tl.int32,
    width: tl.int32,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    b = pid // (channels * height * width)
    c = (pid // (height * width)) % channels
    h = (pid // width) % height
    w = pid % width

    idx_in = b * channels * height * width + c * height * width + h * width + w
    x = tl.load(in_ptr + idx_in)
    idx_sum = b * channels * 1 * width + c * 1 * width + 0 * width + w
    s = tl.load(sum_ptr + idx_sum)
    out = x / s
    tl.store(out_ptr + idx_in, out)

# Wrapper for sum
@torch.fx.wrap
def sum_wrapper(in_1):
    batch, channels, height, width = in_1.shape
    sum_tensor = torch.empty((batch, channels, 1, width), dtype=in_1.dtype, device=in_1.device)
    num_blocks = batch * channels * width
    sum_kernel[(num_blocks,)](
        in_1,
        sum_tensor,
        batch,
        channels,
        height,
        width,
        8
    )
    return sum_tensor

# Wrapper for division
@torch.fx.wrap
def div_wrapper(in_1, sum_tensor):
    batch, channels, height, width = in_1.shape
    result = torch.empty_like(in_1)
    num_blocks = batch * channels * height * width
    div_kernel[(num_blocks,)](
        in_1,
        sum_tensor,
        result,
        batch,
        channels,
        height,
        width,
        8
    )
    return result

# Pattern matching function
def pattern(in_1):
    tmp_0 = in_1.sum(dim=2, keepdim=True)
    tmp_1 = in_1 / tmp_0
    return tmp_1

# Argument extraction

def replacement_args(in_1):
    return (in_1,)

# Replacement function

def normalized(in_1):
    sum_tensor = sum_wrapper(in_1)
    return div_wrapper(in_1, sum_tensor)

def replacement_func():
    return normalized