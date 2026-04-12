import torch
import triton
import triton.language as tl

def pattern(in_2):
    tmp_0 = in_2.sigmoid()
    return tmp_0

def replacement_args(in_2):
    return (in_2,)

@triton.jit
def sigmoid_broadcast_kernel(
    in_ptr,
    out_ptr,
    batch_size,
    channels,
    spatial_size,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < (batch_size * channels * spatial_size)
    
    # Load input data - in_2 is [batch_size, 1, channels]
    flat_in = tl.load(in_ptr + offsets, mask=mask, other=0.0)
    
    # Compute sigmoid
    sigmoid_result = 1.0 / (1.0 + tl.exp(-flat_in))
    
    # For broadcasting: each sigmoid value gets repeated for all spatial positions
    # The output should be [batch_size, channels, spatial_height, spatial_width]
    # We'll compute the sigmoid once and let the caller handle broadcasting
    tl.store(out_ptr + offsets, sigmoid_result, mask=mask)

@torch.fx.wrap
def fused_sigmoid_broadcast(in_2):
    # Get input tensor properties
    batch_size, _, channels = in_2.shape
    spatial_size = 1  # This will be set by the caller based on target shape
    
    # Calculate total elements for output
    total_elements = batch_size * channels * spatial_size
    
    # Choose optimal block size
    BLOCK_SIZE = 1024
    num_programs = (total_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Create output tensor (temporary, will be broadcasted)
    out = torch.empty(in_2.shape, dtype=in_2.dtype, device=in_2.device)
    
    # Launch kernel
    sigmoid_broadcast_kernel[(num_programs,)](
        in_ptr=in_2,
        out_ptr=out,
        batch_size=batch_size,
        channels=channels,
        spatial_size=spatial_size,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out

def replacement_func():
    return lambda in_2: fused_sigmoid_broadcast(in_2)