import torch
import triton
import triton.language as tl

def pattern(in_1):
    tmp_0 = torch.nn.functional.relu(in_1, inplace=True)
    tmp_3 = tmp_0.mean((2, 3), keepdim=True)
    return tmp_0, tmp_3

def replacement_args(in_1):
    return (in_1,)

@triton.jit
def relu_mean_kernel(
    input_ptr,
    output_ptr,
    mean_ptr,
    n_channels,
    height,
    width,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles one channel
    channel_idx = tl.program_id(0)
    
    # Calculate start and end indices for the current channel
    channel_offset = channel_idx * height * width
    program_start = tl.program_id(1) * BLOCK_SIZE
    program_end = min((tl.program_id(1) + 1) * BLOCK_SIZE, height * width)
    
    # Initialize sum for mean computation
    sum_val = 0.0
    
    # Process elements in the channel
    for offset in range(program_start, program_end):
        global_offset = channel_offset + offset
        
        # Load element and apply ReLU
        x = tl.load(input_ptr + global_offset)
        relu_x = tl.maximum(x, 0.0)
        
        # Store ReLU result
        tl.store(output_ptr + global_offset, relu_x)
        
        # Accumulate for mean
        sum_val += relu_x
    
    # Compute mean for this channel block
    if program_end > program_start:
        mean_val = sum_val / (program_end - program_start)
    else:
        mean_val = 0.0
    
    # Store mean result (mean_ptr has shape [n_channels, 1, 1])
    mean_output_offset = channel_idx
    tl.store(mean_ptr + mean_output_offset, mean_val)

@torch.fx.wrap  
def optimized_relu_mean(in_1):
    # Get input dimensions
    n_channels, height, width = in_1.shape[1], in_1.shape[2], in_1.shape[3]
    n_elements = in_1.numel()
    
    # Output tensors
    tmp_0 = torch.empty_like(in_1)
    tmp_3 = torch.empty((in_1.shape[0], n_channels, 1, 1), dtype=in_1.dtype, device=in_1.device)
    
    # Block size for mean computation
    BLOCK_SIZE = 1024
    
    # Grid setup: (num_channels, num_blocks_per_channel)
    num_blocks_per_channel = (height * width + BLOCK_SIZE - 1) // BLOCK_SIZE
    grid = (n_channels, num_blocks_per_channel)
    
    # Launch kernel
    relu_mean_kernel[grid](
        input_ptr=in_1,
        output_ptr=tmp_0,
        mean_ptr=tmp_3.view(-1),  # Flatten for easier indexing
        n_channels=n_channels,
        height=height,
        width=width,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return tmp_0, tmp_3

def replacement_func():
    return optimized_relu_mean