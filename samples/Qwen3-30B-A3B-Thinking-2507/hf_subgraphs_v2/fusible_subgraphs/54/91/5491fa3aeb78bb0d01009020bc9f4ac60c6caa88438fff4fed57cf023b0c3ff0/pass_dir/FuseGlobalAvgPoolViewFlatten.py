import torch
import triton
import triton.language as tl

def pattern(in_0):
    tmp_0 = torch.nn.functional.hardtanh(in_0, 0.0, 6.0, True)
    tmp_1 = torch.nn.functional.adaptive_avg_pool2d(tmp_0, (1, 1))
    tmp_2 = tmp_1.view(4, -1)
    tmp_3 = torch.flatten(tmp_2, 1)
    return tmp_3

def replacement_args(in_0):
    return (in_0,)

@triton.jit
def global_avg_pool_kernel(
    input_ptr,
    output_ptr,
    batch: tl.int32,
    channels: tl.int32,
    H: tl.int32,
    W: tl.int32,
):
    batch_id = tl.program_id(0)
    channel_id = tl.program_id(1)
    input_offset = batch_id * channels * H * W + channel_id * H * W
    output_idx = batch_id * channels + channel_id

    sum_val = 0.0
    for h in range(H):
        for w in range(W):
            input_idx = input_offset + h * W + w
            sum_val += tl.load(input_ptr + input_idx)
    
    avg_val = sum_val / (H * W)
    tl.store(output_ptr + output_idx, avg_val)

@torch.fx.wrap
def global_avg_pool(x):
    batch, channels, H, W = x.shape
    out = torch.empty((batch, channels), dtype=x.dtype, device=x.device)
    grid = (batch, channels)
    global_avg_pool_kernel[grid](
        x,
        out,
        batch,
        channels,
        H,
        W,
    )
    return out

def replacement_func():
    return global_avg_pool