import torch
import triton
import triton.language as tl

def pattern(in_2, in_1):
    """
    Pattern: sigmoid → view(1,-1,1,1) → expand_as
    This matches the gating mechanism used in the computation
    """
    tmp_0 = in_2.sigmoid()
    tmp_1 = tmp_0.view(1, -1, 1, 1)
    tmp_2 = tmp_1.expand_as(in_1)
    return tmp_2

def replacement_args(in_2, in_1):
    return (in_2, in_1)

@triton.jit
def fused_sigmoid_broadcast_kernel(
    sigmoid_ptr,
    out_ptr,
    sigmoid_batch,
    sigmoid_channels,
    target_batch,
    target_channels,
    target_height,
    target_width,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    total_elements = target_batch * target_channels * target_height * target_width
    
    # Each program handles a block of elements
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < total_elements
    
    # Calculate indices
    batch_idx = offsets // (target_channels * target_height * target_width)
    remainder = offsets % (target_channels * target_height * target_width)
    channel_idx = remainder // (target_height * target_width)
    spatial_idx = remainder % (target_height * target_width)
    
    # Load sigmoid value (same for all spatial positions in each channel)
    sigmoid_offset = batch_idx * sigmoid_channels + channel_idx
    sigmoid_val = tl.load(sigmoid_ptr + sigmoid_offset, mask=offsets < sigmoid_batch * sigmoid_channels, other=0.0)
    
    # Apply sigmoid and broadcast to all spatial locations
    sigmoid_out = 1.0 / (1.0 + tl.exp(-sigmoid_val))
    tl.store(out_ptr + offsets, sigmoid_out, mask=mask)

@torch.fx.wrap
def fused_sigmoid_broadcast(in_2, in_1):
    """Fused sigmoid + broadcast operation"""
    # Get target shape from in_1
    target_shape = in_1.shape
    
    # Create output tensor
    out = torch.empty_like(in_1)
    
    # Get dimensions
    batch, _, height, width = target_shape
    channels = in_2.shape[1]  # in_2 shape is [1, 1, 2048] -> channels = 2048
    
    # Calculate total elements
    total_elements = batch * channels * height * width
    
    # Choose block size
    BLOCK_SIZE = 1024
    
    # Calculate grid size
    grid_size = (total_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Launch kernel
    fused_sigmoid_broadcast_kernel[grid_size](
        in_2,  # sigmoid_ptr - we'll compute sigmoid in kernel
        out,
        batch,
        channels,
        target_batch=batch,
        target_channels=channels,
        target_height=height,
        target_width=width,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return out

def replacement_func():
    return fused_sigmoid_broadcast