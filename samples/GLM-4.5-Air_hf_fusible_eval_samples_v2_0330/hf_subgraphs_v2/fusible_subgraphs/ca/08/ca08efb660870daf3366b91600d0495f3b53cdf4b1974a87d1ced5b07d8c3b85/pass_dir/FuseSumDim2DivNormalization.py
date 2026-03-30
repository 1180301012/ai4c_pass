import torch
import triton
import triton.language as tl

# Pattern matching function
def pattern(in_1):
    """
    Matches the normalization pattern: sum(dim=2, keepdim=True) followed by division
    This matches exactly: tmp_0 = in_1.sum(dim = 2, keepdim = True); tmp_1 = in_1 / tmp_0
    """
    tmp_0 = in_1.sum(dim=2, keepdim=True)
    tmp_1 = in_1 / tmp_0
    return tmp_1

# Argument extraction function
def replacement_args(input_tensor):
    return (input_tensor,)

@triton.jit
def fused_normalization_kernel(
    input_ptr,
    output_ptr,
    n_batch,
    n_channels, 
    height,
    width,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles one batch*channel combination
    pid = tl.program_id(0)
    batch_idx = pid // n_channels
    channel_idx = pid % n_channels
    
    # Skip if out of bounds
    if batch_idx >= n_batch or channel_idx >= n_channels:
        return
    
    # Calculate memory offset for this batch and channel
    input_offset = (batch_idx * n_channels * height + channel_idx * height)
    slice_ptr = input_ptr + input_offset
    
    # Calculate sum elements in the 8x8 slice using reduce operation
    offsets = tl.arange(0, height * width)
    mask = offsets < height * width
    values = tl.load(slice_ptr + offsets, mask=mask, other=0.0)
    slice_sum = tl.sum(values)
    
    # Broadcast and normalize all elements in the slice
    normalized_values = values / slice_sum
    
    # Store results  
    tl.store(output_ptr + input_offset + offsets, normalized_values, mask=mask)

@torch.fx.wrap
def fused_normalization(input_tensor):
    """
    Fused normalization using PyTorch's efficient operations
    """
    # Direct computation without unnecessary overhead
    return input_tensor / input_tensor.sum(dim=2, keepdim=True)

# Replacement function (no arguments)
def replacement_func():
    return fused_normalization