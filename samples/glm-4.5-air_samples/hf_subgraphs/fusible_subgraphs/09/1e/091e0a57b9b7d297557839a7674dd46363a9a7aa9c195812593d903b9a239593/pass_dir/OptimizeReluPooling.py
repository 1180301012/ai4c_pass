import torch
import triton
import triton.language as tl

def pattern(prev_output):
    # Pattern: ReLU -> adaptive_avg_pool2d(1) -> flatten
    tmp_5 = torch.nn.functional.relu(prev_output, inplace=True)
    tmp_6 = torch.nn.functional.adaptive_avg_pool2d(tmp_5, 1)
    tmp_7 = tmp_6.flatten(1, -1)
    return tmp_7

def replacement_args(prev_output):
    return (prev_output,)

@triton.jit
def relu_avg_pool_kernel(
    input_ptr,
    output_ptr,
    num_channels,
    height,
    width,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a range of channels
    channel_offset = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    channel_mask = channel_offset < num_channels
    
    # Initialize accumulators for spatial averages
    sums = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    counts = tl.zeros([BLOCK_SIZE], dtype=tl.int32)
    
    # Process spatial dimensions
    total_pixels = height * width
    pixels_per_program = total_pixels
    
    for pixel_idx in range(pixels_per_program):
        # Calculate spatial pixel index
        base_offset = channel_offset[:, None] * total_pixels + pixel_idx
        
        # Load input with bounds checking  
        input_vals = tl.load(input_ptr + base_offset, mask=channel_mask[:, None], other=0.0)
        
        # Apply ReLU and accumulate
        relu_vals = tl.maximum(input_vals, 0.0)
        sums += tl.sum(relu_vals, axis=1)
        counts += tl.sum(channel_mask[:, None], axis=1)
    
    # Compute averages
    count_mask = counts > 0
    averages = tl.where(count_mask, sums / counts.astype(tl.float32), 0.0)
    
    # Store spatial average (one value per channel)
    channel_ptrs = channel_offset
    tl.store(output_ptr + channel_ptrs, averages, mask=channel_mask)

@torch.fx.wrap  
def optimized_relu_avg_pool(prev_output):
    num_channels, height, width = prev_output.shape[1], prev_output.shape[2], prev_output.shape[3]
    total_elements = prev_output.numel()
    
    # Optimal block size for channel processing
    BLOCK_SIZE = min(2048, num_channels)
    num_programs = (num_channels + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Create output tensor - we only need one value per channel
    # This corresponds to the flatten(1, -1) operation
    output_shape = (prev_output.shape[0], num_channels)
    out = torch.empty(output_shape, dtype=torch.float32, device=prev_output.device)
    
    # Launch kernel - each program handles a block of channels
    # We process all spatial pixels within each program
    relu_avg_pool_kernel[(num_programs,)](
        input_ptr=prev_output,
        output_ptr=out,
        num_channels=num_channels,
        height=height,
        width=width,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out

def replacement_func():
    return optimized_relu_avg_pool