import torch
import triton
import triton.language as tl

def pattern(in_0, in_1, in_2, in_3):
    """Match the concatenation + adaptive pooling + dropout + flatten sequence"""
    tmp_0 = torch.cat([in_0, in_1, in_2, in_3], 1)
    tmp_1 = torch.nn.functional.adaptive_avg_pool2d(tmp_0, (1, 1))
    tmp_2 = torch.nn.functional.dropout(tmp_1, 0.5, False, False)
    tmp_3 = torch.flatten(tmp_2, 1)
    return tmp_3

def replacement_args(in_0, in_1, in_2, in_3):
    return (in_0, in_1, in_2, in_3)

@triton.jit
def global_avg_pool_kernel(
    input_ptr,
    output_ptr,
    batch_size,
    channels,
    height,
    width,
):
    """Optimized global average pooling kernel"""
    # Each program handles one channel in one batch
    pid = tl.program_id(0)
    channel_idx = pid % channels
    batch_idx = (pid // channels) % batch_size
    
    # Calculate input pointer offset for this channel and batch
    input_offset = batch_idx * channels * height * width + channel_idx * height * width
    
    # Compute average over spatial dimensions manually
    sum_val = 0.0
    for h in range(height):
        for w in range(width):
            offset = input_offset + h * width + w
            data = tl.load(input_ptr + offset)
            sum_val += data
    
    # Convert to average
    avg_val = sum_val / (height * width)
    
    # Store result
    output_offset = batch_idx * channels + channel_idx
    tl.store(output_ptr + output_offset, avg_val)

@triton.jit
def fused_concat_pool_kernel(
    in0_ptr, in1_ptr, in2_ptr, in3_ptr,
    output_ptr,
    n0_channels, n1_channels, n2_channels, n3_channels,
    batch_size, height, width,
    BLOCK_SIZE: tl.constexpr,
):
    """Fused kernel that concatenates 4 tensors and performs global average pooling"""
    total_channels = n0_channels + n1_channels + n2_channels + n3_channels
    
    # Each program handles one channel across all batches
    pid = tl.program_id(0)
    batch_idx = (pid // total_channels) % batch_size
    channel_idx = pid % total_channels
    
    # Calculate output offset
    output_offset = batch_idx * total_channels + channel_idx
    
    # Initialize sum for global average pooling
    spatial_sum = 0.0
    spatial_elements = height * width
    
    # Determine which input tensor and process spatial elements
    if channel_idx < n0_channels:
        # Input 0
        input_offset = batch_idx * n0_channels * height * width + channel_idx * height * width
        for h in range(height):
            for w in range(width):
                offset = input_offset + h * width + w
                spatial_sum += tl.load(in0_ptr + offset)
    elif channel_idx < n0_channels + n1_channels:
        # Input 1
        local_idx = channel_idx - n0_channels
        input_offset = batch_idx * n1_channels * height * width + local_idx * height * width
        for h in range(height):
            for w in range(width):
                offset = input_offset + h * width + w
                spatial_sum += tl.load(in1_ptr + offset)
    elif channel_idx < n0_channels + n1_channels + n2_channels:
        # Input 2
        local_idx = channel_idx - n0_channels - n1_channels
        input_offset = batch_idx * n2_channels * height * width + local_idx * height * width
        for h in range(height):
            for w in range(width):
                offset = input_offset + h * width + w
                spatial_sum += tl.load(in2_ptr + offset)
    else:
        # Input 3
        local_idx = channel_idx - n0_channels - n1_channels - n2_channels
        input_offset = batch_idx * n3_channels * height * width + local_idx * height * width
        for h in range(height):
            for w in range(width):
                offset = input_offset + h * width + w
                spatial_sum += tl.load(in3_ptr + offset)
    
    # Compute average and store
    avg_val = spatial_sum / spatial_elements
    tl.store(output_ptr + output_offset, avg_val)

@torch.fx.wrap  
def optimized_global_pooling(in_0, in_1, in_2, in_3):
    """Optimized fused concatenation and global average pooling"""
    # Get tensor dimensions
    batch_size, c0, height, width = in_0.shape
    _, c1, _, _ = in_1.shape
    _, c2, _, _ = in_2.shape
    _, c3, _, _ = in_3.shape
    total_channels = c0 + c1 + c2 + c3
    
    # Allocate output tensor (batch_size, total_channels)
    output = torch.empty(batch_size, total_channels, dtype=in_0.dtype, device=in_0.device)
    
    # Launch fused kernel
    BLOCK_SIZE = 1024
    total_elements = batch_size * total_channels
    grid = (total_elements,)
    
    fused_concat_pool_kernel[grid](
        in0_ptr=in_0,
        in1_ptr=in_1,
        in2_ptr=in_2,
        in3_ptr=in_3,
        output_ptr=output,
        n0_channels=c0,
        n1_channels=c1,
        n2_channels=c2,
        n3_channels=c3,
        batch_size=batch_size,
        height=height,
        width=width,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return output

def replacement_func():
    return optimized_global_pooling