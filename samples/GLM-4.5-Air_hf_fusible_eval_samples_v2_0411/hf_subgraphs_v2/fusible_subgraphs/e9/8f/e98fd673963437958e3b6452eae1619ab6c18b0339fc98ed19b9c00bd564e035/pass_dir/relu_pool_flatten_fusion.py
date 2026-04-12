import torch
import triton
import triton.language as tl

def pattern(tmp_4):
    tmp_5 = torch.nn.functional.relu(tmp_4, inplace = True);  tmp_4 = None
    tmp_6 = torch.nn.functional.adaptive_avg_pool2d(tmp_5, 1);  tmp_5 = None
    tmp_7 = tmp_6.flatten(1, -1);  tmp_6 = None
    return tmp_7

def replacement_args(tmp_4):
    return (tmp_4,)

@triton.jit
def relu_pool_flatten_kernel(
    input_ptr,
    output_ptr,
    n_channels,
    height,
    width,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_channels  # One element per channel
    
    # We need to process each channel's spatial data, but we output one value per channel
    # Each program handles one channel
    
    # For the current channel, calculate its spatial location in memory
    channel_idx = offsets
    
    # Since we're doing global average pooling followed by flattening,
    # we need to compute the mean of each channel across all spatial positions
    
    # For each channel, we need to sum all its spatial elements and then divide by count
    channel_sum = 0.0
    element_count = height * width
    
    # For each spatial position in this channel's data
    for h in range(height):
        for w in range(width):
            # Calculate the global memory offset for this spatial position
            # Input layout: [1, n_channels, height, width]
            # Memory layout: spatial-major within each channel
            global_offset = (h * width + w) * n_channels + channel_idx
            
            # Load element with bounds checking
            spatial_mask = global_offset < (n_channels * height * width)
            element = tl.load(input_ptr + global_offset, mask=spatial_mask, other=0.0)
            
            # Apply ReLU and accumulate
            channel_sum = channel_sum + tl.maximum(element, 0.0)
    
    # Compute average for this channel
    channel_mean = channel_sum / element_count
    
    # Store the result (one value per channel)
    tl.store(output_ptr + channel_idx, channel_mean, mask=mask)

@torch.fx.wrap
def fused_relu_pool_flatten(input_tensor):
    # Get input tensor properties
    batch_size, n_channels, height, width = input_tensor.shape
    
    # Output should be [batch_size, n_channels]
    output_size = n_channels
    batch_size = input_tensor.shape[0]
    
    # Choose block size
    BLOCK_SIZE = min(1024, n_channels)  # One element per channel in output
    num_programs = (n_channels + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Create output tensor
    output = torch.empty(batch_size, n_channels, dtype=input_tensor.dtype, device=input_tensor.device)
    
    # Launch kernel - expand batch dimension appropriately
    # For simplicity, process first batch element (assume batch_size = 1 based on weight_meta)
    relu_pool_flatten_kernel[(num_programs, 1)](
        input_ptr=input_tensor,
        output_ptr=output[0],  # Process first tensor in batch
        n_channels=n_channels,
        height=height,
        width=width,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    # Handle other batch elements if needed (but weight_meta shows batch_size=1)
    if batch_size > 1:
        for i in range(1, batch_size):
            relu_pool_flatten_kernel[(num_programs, 1)](
                input_ptr=input_tensor[i],
                output_ptr=output[i],
                n_channels=n_channels,
                height=height,
                width=width,
                BLOCK_SIZE=BLOCK_SIZE,
            )
    
    return output

def replacement_func():
    return fused_relu_pool_flatten