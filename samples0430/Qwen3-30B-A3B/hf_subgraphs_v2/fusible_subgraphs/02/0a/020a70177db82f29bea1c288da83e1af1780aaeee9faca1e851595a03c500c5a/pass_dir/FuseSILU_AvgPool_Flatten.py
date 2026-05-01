import torch
import triton
import triton.language as tl

def pattern(in_0):
    tmp_0 = torch.nn.functional.silu(in_0, inplace=True)
    tmp_1 = torch.nn.functional.adaptive_avg_pool2d(tmp_0, 1)
    tmp_2 = torch.flatten(tmp_1, 1)
    return tmp_2

def replacement_args(in_0):
    return (in_0,)

@triton.jit
def fused_kernel(
    input_ptr,
    output_ptr,
    batch_size,
    channels,
    height,
    width,
    BLOCK_SIZE: tl.constexpr = 256
):
    pid = tl.program_id(0)
    start_index = pid * BLOCK_SIZE
    indices = start_index + tl.arange(0, BLOCK_SIZE)
    
    batch_mask = indices // channels < batch_size
    channel_mask = indices % channels < channels
    valid = batch_mask & channel_mask

    for i in range(BLOCK_SIZE):
        if not valid[i]:
            continue
        b = indices[i] // channels
        c = indices[i] % channels

        total = 0.0
        for h in range(height):
            for w in range(width):
                idx = b * (channels * height * width) + c * (height * width) + h * width + w
                val = tl.load(input_ptr + idx, mask=valid[i], other=0.0)
                total += val
        mean_val = total / (height * width)
        sigmoid_val = 1.0 / (1.0 + tl.exp(-mean_val))
        silu_val = mean_val * sigmoid_val
        out_idx = b * channels + c
        tl.store(output_ptr + out_idx, silu_val, mask=valid[i])

@torch.fx.wrap
def fused_silu_pool_flatten(in_0):
    in_0_fp32 = in_0.to(torch.float32)
    batch_size, channels, height, width = in_0_fp32.shape
    out_fp32 = torch.empty((batch_size, channels), dtype=torch.float32, device=in_0_fp32.device)
    
    BLOCK_SIZE = 256
    num_blocks = (batch_size * channels + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    fused_kernel[(num_blocks,)](
        input_ptr=in_0_fp32,
        output_ptr=out_fp32,
        batch_size=batch_size,
        channels=channels,
        height=height,
        width=width,
        BLOCK_SIZE=BLOCK_SIZE
    )
    out = out_fp32.to(in_0.dtype)
    return out

def replacement_func():
    return fused_silu_pool_flatten