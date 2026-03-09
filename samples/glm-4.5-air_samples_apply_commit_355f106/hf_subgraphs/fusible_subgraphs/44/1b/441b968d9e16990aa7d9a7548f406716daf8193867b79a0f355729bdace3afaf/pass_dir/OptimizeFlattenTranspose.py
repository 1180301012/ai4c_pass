import torch
import triton
import triton.language as tl

@triton.jit
def flatten_transpose_kernel(
    input_ptr,
    output_ptr,
    batch_size,
    h,
    w, 
    channels,
    BLOCK_SIZE_H: tl.constexpr,
):
    batch_idx = tl.program_id(0)
    channel_idx = tl.program_id(1)
    h_idx = tl.program_id(2) * BLOCK_SIZE_H + tl.arange(0, BLOCK_SIZE_H)
    
    mask = h_idx < h
    
    # Load input: [batch, channels, h, w]
    # Flatten(2) -> [batch, channels, h*w]
    # Transpose(1, 2) -> [batch, h*w, channels]
    
    input_val = tl.load(input_ptr + batch_idx * channels * h * w + channel_idx * h * w + h_idx * w, mask=mask, other=0.0)
    
    # Map to output position: [batch, h_idx*w + channel_idx, hidden_size]
    output_pos = batch_idx * (h * w * channels) + h_idx * w + channel_idx
    output_flat_idx = batch_idx * (h * w * channels) + h_idx * channels + channel_idx * h
    
    tl.store(output_ptr + output_flat_idx, input_val, mask=mask)

@torch.fx.wrap  
def optimized_flatten_transpose(x):
    batch_size, channels, h, w = x.shape
    out = torch.empty(batch_size, h * w, channels, dtype=x.dtype, device=x.device)
    
    BLOCK_SIZE_H = 64
    grid = (
        batch_size,
        channels,
        (h + BLOCK_SIZE_H - 1) // BLOCK_SIZE_H
    )
    
    flatten_transpose_kernel[grid](
        x, out,
        batch_size, h, w, channels,
        BLOCK_SIZE_H
    )
    
    return out

def pattern(x):
    tmp_7 = x.flatten(2)
    tmp_8 = tmp_7.transpose(1, 2)
    return tmp_8

def replacement_args(x):
    return (x,)

def replacement_func():
    return optimized_flatten_transpose