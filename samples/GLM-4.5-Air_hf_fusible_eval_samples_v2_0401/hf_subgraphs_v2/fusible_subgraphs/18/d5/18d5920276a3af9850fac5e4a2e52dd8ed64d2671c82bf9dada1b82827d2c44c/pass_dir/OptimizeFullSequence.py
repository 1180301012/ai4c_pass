import torch
import triton
import triton.language as tl

def pattern(x):
    """
    Matches the full sequence: ReLU -> Dropout(p=0.0) -> Flatten
    This eliminates the no-op dropout and fuses ReLU with flatten
    """
    tmp_0 = torch.nn.functional.relu(x, inplace=False)
    tmp_1 = torch.nn.functional.dropout(tmp_0, 0.0, False, False)
    tmp_2 = tmp_1.flatten(1, -1)
    return tmp_2  # Return the final output to match the observable result

def replacement_args(x):
    """Extract arguments for the replacement"""
    return (x,)

@triton.jit
def full_sequence_kernel(
    x_ptr, 
    out_ptr, 
    batch_size, 
    channels,
    BLOCK_SIZE: tl.constexpr
):
    """
    Complete optimized kernel: ReLU (eliminating dropout) + Flatten
    Handles input shape [batch_size, channels, 1, 1] -> [batch_size, channels]
    """
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < (batch_size * channels)
    
    # Reshape linear offset to [batch_idx, channel_idx] for the original tensor
    channel_idx = offsets % channels
    batch_idx = offsets // channels
    
    # Calculate original tensor offset ([batch, channels, 1, 1] layout)
    orig_offset = batch_idx * channels + channel_idx
    
    # Load input data, apply ReLU directly (dropout p=0.0 is eliminated)
    x = tl.load(x_ptr + orig_offset, mask=mask, other=0.0)
    out = tl.maximum(x, 0.0)
    
    # Store directly to flattened output [batch_size, channels]
    tl.store(out_ptr + offsets, out, mask=mask)

@torch.fx.wrap
def optimized_full_sequence(x):
    """
    Fully optimized sequence: eliminates no-op dropout and fuses ReLU with flatten
    """
    input_shape = x.shape
    batch_size, channels = input_shape[0], input_shape[1]
    
    # Output shape: [batch_size, channels] after flatten(1, -1)
    out_shape = [batch_size, channels]
    out = torch.empty(out_shape, dtype=x.dtype, device=x.device)
    
    N = batch_size * channels
    BLOCK_SIZE = 1024
    num_programs = (N + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    full_sequence_kernel[(num_programs,)](
        x_ptr=x,
        out_ptr=out,
        batch_size=batch_size,
        channels=channels,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out

def replacement_func():
    """Return the optimized function"""
    return optimized_full_sequence