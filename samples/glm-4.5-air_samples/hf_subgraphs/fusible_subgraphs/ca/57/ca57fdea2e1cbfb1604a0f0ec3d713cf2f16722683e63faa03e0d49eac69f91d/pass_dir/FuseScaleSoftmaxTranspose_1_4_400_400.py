import torch
import triton
import triton.language as tl
import math

def pattern(x):
    # Match the computation pattern:
    # 1. Scale by constant
    # 2. Apply softmax along last dimension
    # 3. Transpose last two dimensions
    tmp_0 = x * 0.1767766952966369
    tmp_1 = tmp_0.softmax(dim=-1)
    tmp_2 = tmp_1.transpose(-2, -1)
    return tmp_2

def replacement_args(x):
    return (x,)

@triton.heuristics({"BLOCK_SIZE": lambda args: min(args["width"], 128)})
@triton.jit
def scale_softmax_transpose_kernel(
    x_ptr,
    out_ptr,
    batch_size,
    channels,
    height,
    width,
    scale_factor: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a contiguous block along the softmax dimension (width)
    pid = tl.program_id(0)
    
    # Calculate which batch, channel, height this program handles
    batch_size_channels_height = batch_size * channels * height
    batch_channel_idx = pid // height
    height_idx = pid % height
    
    batch_idx = batch_channel_idx // channels
    channel_idx = batch_channel_idx % channels
    
    # Process a block of width elements
    start_w = 0
    end_w = width
    
    offsets = start_w + tl.arange(0, BLOCK_SIZE)
    mask = offsets < end_w
    
    # Load the current row (softmax input along width dimension)
    x_base_ptr = x_ptr + batch_idx * channels * height * width + channel_idx * height * width
    x_row = tl.load(x_base_ptr + height_idx * width + offsets, mask=mask, other=float('-inf'))
    
    # Scale the input values
    scaled_x = x_row * scale_factor
    
    # Compute softmax along the width dimension
    # Use vectorized operations - for partially filled vectors,
    # we need to be careful about edge cases
    
    # Find maximum value for numerical stability
    max_val = tl.max(scaled_x)
    
    # Compute exponentials
    exp_x = tl.exp(scaled_x - max_val)
    
    # Compute sum of exponentials
    sum_exp = tl.sum(exp_x)
    
    # Compute softmax
    softmax_out = exp_x / sum_exp
    
    # Store result with simpler transpose logic
    # Store softmax result first (without transpose in this simplified version)
    tl.store(out_ptr + height_idx * width + offsets, softmax_out, mask=mask)

@torch.fx.wrap  
def fused_scale_softmax_transpose(x):
    # Get input dimensions
    batch_size, channels, height, width = x.shape
    
    # Use efficient Triton implementation for all sizes with adaptive optimization
    out = torch.empty_like(x)
    
    # Smart block size selection based on workload characteristics
    if batch_size == 1:
        BLOCK_SIZE = min(64, width)  # Smaller block for single batch to reduce overhead
    else:
        BLOCK_SIZE = min(128, width)  # Larger block for batch processing
    
    # Total number of programs needed (batch * channels * height)
    total_programs = batch_size * channels * height
    
    # Calculate number of programs needed
    num_programs = (total_programs + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Launch the optimized fused kernel
    scale_softmax_transpose_kernel[(num_programs,)](
        x_ptr=x,
        out_ptr=out,
        batch_size=batch_size,
        channels=channels,
        height=height,
        width=width,
        scale_factor=0.1767766952966369,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    # Apply transpose efficiently as separate operation
    result = out.transpose(-2, -1)
    return result

def replacement_func():
    return fused_scale_softmax_transpose