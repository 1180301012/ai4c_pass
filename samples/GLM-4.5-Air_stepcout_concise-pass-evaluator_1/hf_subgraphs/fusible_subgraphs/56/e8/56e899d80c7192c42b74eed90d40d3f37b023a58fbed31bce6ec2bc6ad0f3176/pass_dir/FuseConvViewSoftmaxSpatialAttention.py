import torch
import triton
import triton.language as tl

def pattern(in_0, in_1, in_2):
    """Match the conv2d + view + softmax pattern for spatial attention mechanism"""
    tmp_0 = in_0
    tmp_1 = in_1
    tmp_2 = torch.conv2d(in_2, tmp_1, tmp_0, (1, 1), (0, 0), (1, 1), 1)
    tmp_3 = tmp_2.view(1, 1, -1)  # Will be adjusted in fused function based on actual batch size
    tmp_4 = tmp_3.softmax(dim=-1)
    return (tmp_4,)

def replacement_args(in_0, in_1, in_2):
    """Extract arguments needed for fused kernel"""
    return (in_0, in_1, in_2)

@triton.jit
def spatial_attention_kernel(
    input_ptr,
    weight_ptr,
    bias_ptr,
    output_ptr,
    batch_size,
    num_input_channels,
    height,
    width,
    BLOCK_SIZE: tl.constexpr,
):
    """Fused kernel for conv2d + view + softmax spatial attention"""
    
    # Each program handles one spatial location across all batches and channels
    program_id = tl.program_id(0)
    spatial_size = height * width
    total_elements = batch_size * spatial_size
    
    if program_id >= total_elements:
        return
    
    # Calculate batch and spatial indices
    batch_idx = program_id // spatial_size
    spatial_idx = program_id % spatial_size
    h = spatial_idx // width
    w = spatial_idx % width
    
    # Batch pointer offset for this location: [batch_idx, :, h, w]
    batch_offset = batch_idx * num_input_channels * height * width
    
    # Efficient 1x1 convolution: weighted sum of channels at spatial location
    conv_result = 0.0
    for c in range(num_input_channels):
        input_offset = batch_offset + c * height * width + h * width + w
        input_val = tl.load(input_ptr + input_offset)
        
        # Weight offset for channel c in [1, num_input_channels, 1, 1]
        weight_offset = c
        weight_val = tl.load(weight_ptr + weight_offset)
        
        conv_result += input_val * weight_val
    
    # Add bias
    bias_val = tl.load(bias_ptr)
    conv_result += bias_val
    
    # Apply softmax across the flattened spatial dimension for each batch
    # We need to compute exp(conv_result) and normalize by sum of exp across spatial dimension
    
    # Calculate starting index for this batch in the flattened output
    batch_start = batch_idx * spatial_size
    
    # Reduction for softmax: sum of exp values across spatial dimension
    # Note: This is simplified - in practice we'd need a separate reduction pass
    # For this optimized version, we'll compute softmax using a more efficient approach
    
    # Store the conv result (pre-softmax)
    # Output has shape [batch_size, 1, height, width]
    output_batch_offset = batch_idx * (1 * height * width)
    output_channel_offset = 0  # channels = 1
    output_offset = output_batch_offset + output_channel_offset * (height * width) + h * width + w
    tl.store(output_ptr + output_offset, conv_result)

@triton.jit
def softmax_max_kernel(
    input_ptr,
    max_ptr,
    batch_size,
    spatial_size,
    BLOCK_SIZE: tl.constexpr,
):
    """Find max values for each batch for softmax numerical stability"""
    program_id = tl.program_id(0)
    
    if program_id >= batch_size:
        return
    
    # Initialize max value for this batch
    max_val = -float('inf')
    
    # Load all data for this batch using scalar loads
    batch_offset = program_id * spatial_size
    for i in range(spatial_size):
        val = tl.load(input_ptr + batch_offset + i)
        if val > max_val:
            max_val = val
    
    # Store max for this batch
    tl.store(max_ptr + program_id, max_val)

@triton.jit
def softmax_exp_sum_kernel(
    input_ptr,
    max_ptr,
    exp_sum_ptr,
    batch_size,
    spatial_size,
    BLOCK_SIZE: tl.constexpr,
):
    """Compute sum of exp(x - max) for each batch"""
    program_id = tl.program_id(0)
    
    if program_id >= batch_size:
        return
    
    # Load max for this batch
    max_val = tl.load(max_ptr + program_id)
    
    # Compute exp sum for this batch
    exp_sum = 0.0
    batch_offset = program_id * spatial_size
    
    # Use scalar loads for all elements
    for i in range(spatial_size):
        val = tl.load(input_ptr + batch_offset + i)
        exp_val = tl.exp(val - max_val)
        exp_sum += exp_val
    
    # Store exp sum for this batch
    tl.store(exp_sum_ptr + program_id, exp_sum)

@triton.jit
def softmax_final_kernel(
    input_ptr,
    max_ptr,
    exp_sum_ptr,
    output_ptr,
    batch_size,
    spatial_size,
    BLOCK_SIZE: tl.constexpr,
):
    """Compute final softmax values: exp(x - max) / sum"""
    program_id = tl.program_id(0)
    total_elements = batch_size * spatial_size
    
    if program_id >= total_elements:
        return
    
    batch_idx = program_id // spatial_size
    spatial_idx = program_id % spatial_size
    
    # Load max and exp sum for this batch
    max_val = tl.load(max_ptr + batch_idx)
    exp_sum = tl.load(exp_sum_ptr + batch_idx)
    
    # Load input value
    input_offset = batch_idx * spatial_size + spatial_idx
    input_val = tl.load(input_ptr + input_offset)
    
    # Compute softmax
    exp_val = tl.exp(input_val - max_val)
    softmax_val = exp_val / exp_sum
    
    # Store result
    tl.store(output_ptr + input_offset, softmax_val)

def apply_softmax_with_triton(input_tensor, batch_size, spatial_size):
    """Apply softmax across spatial dimension using Triton kernels"""
    
    # Create intermediate tensors for max and exp_sum
    max_vals = torch.zeros(batch_size, dtype=input_tensor.dtype, device=input_tensor.device)
    exp_sums = torch.zeros(batch_size, dtype=input_tensor.dtype, device=input_tensor.device)
    
    block_size = 1024
    
    # Step 1: Find max values for each batch
    num_programs_max = (batch_size + block_size - 1) // block_size
    softmax_max_kernel[(num_programs_max,)](
        input_ptr=input_tensor,
        max_ptr=max_vals,
        batch_size=batch_size,
        spatial_size=spatial_size,
        BLOCK_SIZE=block_size
    )
    
    # Step 2: Compute sum of exp(x - max) for each batch
    num_programs_sum = (batch_size + block_size - 1) // block_size  
    softmax_exp_sum_kernel[(num_programs_sum,)](
        input_ptr=input_tensor,
        max_ptr=max_vals,
        exp_sum_ptr=exp_sums,
        batch_size=batch_size,
        spatial_size=spatial_size,
        BLOCK_SIZE=block_size
    )
    
    # Step 3: Compute final softmax values
    softmax_output = torch.zeros_like(input_tensor)
    num_programs_softmax = (batch_size * spatial_size + block_size - 1) // block_size
    softmax_final_kernel[(num_programs_softmax,)](
        input_ptr=input_tensor,
        max_ptr=max_vals,
        exp_sum_ptr=exp_sums,
        output_ptr=softmax_output,
        batch_size=batch_size,
        spatial_size=spatial_size,
        BLOCK_SIZE=block_size
    )
    
    # Reshape to match original output format: [batch_size, 1, spatial_size]
    return softmax_output.view(batch_size, 1, spatial_size)

@torch.fx.wrap
def fused_spatial_attention(bias, weight, input_tensor):
    """Fused spatial attention with conv2d + view + softmax"""
    
    # Get tensor shapes
    batch_size = input_tensor.shape[0]
    num_input_channels = input_tensor.shape[1] 
    height = input_tensor.shape[2]
    width = input_tensor.shape[3]
    
    # Flattened spatial dimension
    spatial_size = height * width
    
    # Compute standard conv2d using Triton with correct indexing
    conv_output = torch.zeros(batch_size, 1, height, width, dtype=input_tensor.dtype, device=input_tensor.device)
    
    # Launch the fused kernel - each thread handles one spatial location per batch
    block_size = 1024
    total_elements = batch_size * height * width
    num_programs = (total_elements + block_size - 1) // block_size
    
    spatial_attention_kernel[(num_programs,)](
        input_ptr=input_tensor,
        weight_ptr=weight.view(-1),  # Flatten weight to [num_input_channels]
        bias_ptr=bias,
        output_ptr=conv_output,
        batch_size=batch_size,
        num_input_channels=num_input_channels, 
        height=height,
        width=width,
        BLOCK_SIZE=block_size
    )
    
    # Apply view transformation: [batch_size, 1, height, width] -> [batch_size, 1, height*width]
    conv_output_viewed = conv_output.view(batch_size, 1, spatial_size)
    
    # Apply softmax across the last dimension
    softmax_output = apply_softmax_with_triton(conv_output_viewed, batch_size, spatial_size)
    
    return softmax_output

def replacement_func():
    """Return the fused spatial attention kernel"""
    return fused_spatial_attention