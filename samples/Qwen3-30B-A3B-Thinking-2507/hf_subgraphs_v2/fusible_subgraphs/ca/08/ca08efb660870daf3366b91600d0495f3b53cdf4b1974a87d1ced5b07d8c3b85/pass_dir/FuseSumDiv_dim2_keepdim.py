import torch
import triton
import triton.language as tl


def pattern(x):
    sum_out = x.sum(dim=2, keepdim=True)
    out = x / sum_out
    return out

def replacement_args(x):
    return (x, )


@triton.jit
def sum_kernel(
    x_ptr,
    sum_out_ptr,
    batch: tl.int32,
    channels: tl.int32,
    height: tl.int32,
    width: tl.int32,
):
    batch_idx = tl.program_id(0)
    channel_idx = tl.program_id(1)
    width_idx = tl.program_id(2)
    sum_val = tl.zeros((1,), dtype=tl.float16)
    height_idx = tl.thread_id(0)
    if height_idx < height:
        x_idx = batch_idx * channels * height * width + channel_idx * height * width + height_idx * width + width_idx
        x_val = tl.load(x_ptr + x_idx)
        sum_val = x_val
    sum_val = tl.sum(sum_val, axis=0)
    sum_out_idx = batch_idx * channels * width + channel_idx * width + width_idx
    tl.store(sum_out_ptr + sum_out_idx, sum_val)


@triton.jit
def div_kernel(
    x_ptr,
    sum_out_ptr,
    out_ptr,
    batch: tl.int32,
    channels: tl.int32,
    height: tl.int32,
    width: tl.int32,
    BLOCK_SIZE: tl.constexpr,
):
    idx = tl.program_id(0) * BLOCK_SIZE + tl.thread_id(0)
    if idx >= batch * channels * height * width:
        return
    batch_idx = idx // (channels * height * width)
    channel_idx = (idx // (height * width)) % channels
    height_idx = (idx // width) % height
    width_idx = idx % width
    sum_idx = (batch_idx * channels + channel_idx) * width + width_idx
    x_val = tl.load(x_ptr + idx)
    sum_val = tl.load(sum_out_ptr + sum_idx)
    out_val = x_val / sum_val
    tl.store(out_ptr + idx, out_val)


@torch.fx.wrap
def normalize(x):
    batch = x.shape[0]
    channels = x.shape[1]
    height = x.shape[2]
    width = x.shape[3]
    sum_out = torch.empty((batch * channels, width), dtype=x.dtype, device=x.device)
    out = torch.empty_like(x)
    sum_kernel[(batch, channels, width)](x, sum_out, batch, channels, height, width)
    n_elements = batch * channels * height * width
    grid_size = (n_elements + 1023) // 1024
    div_kernel[(grid_size, 1024)](x, sum_out, out, batch, channels, height, width, 1024)
    return out

def replacement_func():
    return normalize