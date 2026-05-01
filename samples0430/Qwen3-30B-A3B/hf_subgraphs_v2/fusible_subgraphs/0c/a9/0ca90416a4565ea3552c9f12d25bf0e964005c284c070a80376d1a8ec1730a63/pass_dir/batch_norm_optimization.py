import torch
import triton
import triton.language as tl

BLOCK_SIZE = 128

def pattern(in_4, in_0, in_1, in_3, in_2):
    return torch.nn.functional.batch_norm(in_4, in_0, in_1, in_3, in_2, False, 0.1, 0.001)

def replacement_args(in_4, in_0, in_1, in_3, in_2):
    return (in_4, in_0, in_1, in_3, in_2)

@triton.jit
def batch_norm_kernel(
    x_ptr, 
    mean_ptr, 
    var_ptr, 
    weight_ptr, 
    bias_ptr, 
    out_ptr,
    channels, 
    height, 
    width,
    BLOCK_SIZE: tl.constexpr = 128
):
    # Process one channel at a time
    channel_idx = tl.program_id(0)
    
    # Compute start index for the current channel
    start_idx = channel_idx * height * width
    
    # Each thread handles a block of data within the channel
    block_start = tl.program_id(1) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < height * width
    
    # Load channel stats
    mean_val = tl.load(mean_ptr + channel_idx)
    var_val = tl.load(var_ptr + channel_idx)
    weight_val = tl.load(weight_ptr + channel_idx)
    bias_val = tl.load(bias_ptr + channel_idx)
    
    # Load input values
    input_vals = tl.load(x_ptr + start_idx + offsets, mask=mask)
    
    # Convert all values to float32 for the computation
    input_vals_float = tl.cast(input_vals, tl.float32)
    mean_val_float = tl.cast(mean_val, tl.float32)
    var_val_float = tl.cast(var_val, tl.float32)
    weight_val_float = tl.cast(weight_val, tl.float32)
    bias_val_float = tl.cast(bias_val, tl.float32)
    
    # Normalize with the correct epsilon
    normalized = (input_vals_float - mean_val_float) * tl.rsqrt(var_val_float + 0.001)
    
    # Scale and add bias
    output_vals_float = normalized * weight_val_float + bias_val_float
    output_vals = tl.cast(output_vals_float, tl.bfloat16)
    
    # Store
    tl.store(out_ptr + start_idx + offsets, output_vals, mask=mask)

@torch.fx.wrap
def custom_batch_norm(x, mean, var, weight, bias):
    batch, channels, height, width = x.shape
    num_channels = channels
    num_blocks = (height * width + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Allocate output tensor
    out = torch.empty_like(x)
    
    # Launch kernel
    batch_norm_kernel[(num_channels, num_blocks)](
        x_ptr=x,
        mean_ptr=mean,
        var_ptr=var,
        weight_ptr=weight,
        bias_ptr=bias,
        out_ptr=out,
        channels=channels,
        height=height,
        width=width,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return out

def replacement_func():
    return custom_batch_norm