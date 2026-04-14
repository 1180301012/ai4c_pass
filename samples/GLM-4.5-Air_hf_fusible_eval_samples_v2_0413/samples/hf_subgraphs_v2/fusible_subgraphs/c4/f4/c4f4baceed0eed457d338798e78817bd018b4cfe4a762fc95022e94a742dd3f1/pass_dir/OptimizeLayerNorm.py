import torch
import triton
import triton.language as tl

def pattern(tmp_8, in_1, in_0):
    tmp_9 = torch.nn.functional.layer_norm(tmp_8, (384,), in_1, in_0, 1e-05)
    return tmp_8, tmp_9

def replacement_args(tmp_8, in_1, in_0):
    return (tmp_8, in_1, in_0)

@triton.jit
def layer_norm_kernel(
    x_ptr,
    weight_ptr,
    bias_ptr,
    y_ptr,
    n_elements: tl.constexpr,
    channels: tl.constexpr,
    eps: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles one row/channel
    row_idx = tl.program_id(0)
    
    # offset within this row
    offsets = row_idx * channels + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load x, weight, and bias
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    weight = tl.load(weight_ptr + offsets, mask=mask, other=1.0)
    bias = tl.load(bias_ptr + offsets, mask=mask, other=0.0)
    
    # Compute mean
    x_sum = tl.sum(x, axis=0)
    x_mean = x_sum / channels
    
    # Compute variance
    x_centered = x - x_mean
    x_var = tl.sum(x_centered * x_centered, axis=0) / channels
    
    # Normalize
    x_norm = (x - x_mean) / tl.sqrt(x_var + eps)
    
    # Scale and shift
    y = x_norm * weight + bias
    
    # Store result
    tl.store(y_ptr + offsets, y, mask=mask)

@triton.jit
def layer_norm_kernel_optimized(
    x_ptr,
    weight_ptr,
    bias_ptr,
    y_ptr,
    n_elements: tl.constexpr,
    channels: tl.constexpr,
    eps: tl.constexpr,
    BLOCK_SIZE_CHANNELS: tl.constexpr,
):
    # Each program handles one row in the sequence
    program_id = tl.program_id(0)
    
    # Process multiple channels per program for better utilization
    channel_offsets = program_id * BLOCK_SIZE_CHANNELS + tl.arange(0, BLOCK_SIZE_CHANNELS)
    mask = channel_offsets < channels
    
    # Load weight and bias once per program
    weight = tl.load(weight_ptr + channel_offsets, mask=mask, other=1.0)
    bias = tl.load(bias_ptr + channel_offsets, mask=mask, other=0.0)
    
    # Load one row of x at a time (all channels)
    x_row = tl.load(x_ptr + program_id * channels + channel_offsets, mask=mask, other=0.0)
    
    # Compute mean using reduction within the program
    row_sum = tl.sum(x_row, axis=0)
    row_mean = row_sum / channels
    
    # Compute variance
    x_centered = x_row - row_mean
    row_var = tl.sum(x_centered * x_centered, axis=0) / channels
    
    # Normalize
    x_norm = (x_row - row_mean) / tl.sqrt(row_var + eps)
    
    # Apply weight and bias
    y_row = x_norm * weight + bias
    
    # Store result
    tl.store(y_ptr + program_id * channels + channel_offsets, y_row, mask=mask)

@torch.fx.wrap
def optimized_layer_norm(x, weight, bias, eps=1e-05):
    # Get tensor dimensions
    if x.dim() == 3:  # [1, seq_len, channels]
        batch_size, seq_len, channels = x.shape
        n_elements = seq_len * channels
    else:  # [seq_len, channels] 
        seq_len, channels = x.shape
        batch_size = 1
        n_elements = seq_len * channels
    
    # Output tensor
    y = torch.empty_like(x)
    
    # Optimize block size based on channel count
    if channels <= 96:
        BLOCK_SIZE_CHANNELS = 32
    elif channels <= 192:
        BLOCK_SIZE_CHANNELS = 64
    else:  # 384
        BLOCK_SIZE_CHANNELS = 128
    
    num_programs = (channels + BLOCK_SIZE_CHANNELS - 1) // BLOCK_SIZE_CHANNELS
    
    # Use optimized kernel
    if channels <= 384:  # Most cases we handle
        layer_norm_kernel_optimized[(num_programs,)](
            x_ptr=x,
            weight_ptr=weight,
            bias_ptr=bias,
            y_ptr=y,
            n_elements=n_elements,
            channels=channels,
            eps=eps,
            BLOCK_SIZE_CHANNELS=BLOCK_SIZE_CHANNELS,
        )
    else:
        # Fallback to original kernel for very large channels
        BLOCK_SIZE = 1024
        num_programs_full = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
        layer_norm_kernel[(num_programs_full,)](
            x_ptr=x,
            weight_ptr=weight,
            bias_ptr=bias,
            y_ptr=y,
            n_elements=n_elements,
            channels=channels,
            eps=eps,
            BLOCK_SIZE=BLOCK_SIZE,
        )
    
    return x, y

def replacement_func():
    return optimized_layer_norm