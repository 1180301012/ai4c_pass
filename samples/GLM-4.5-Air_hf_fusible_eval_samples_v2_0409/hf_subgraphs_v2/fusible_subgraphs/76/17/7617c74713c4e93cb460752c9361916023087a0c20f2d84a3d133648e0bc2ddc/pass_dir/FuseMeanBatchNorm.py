import torch
import triton
import triton.language as tl

def pattern(input_4, input_5, in_0, in_1, in_2, in_3):
    """
    Pattern to match the entire sequence after dropout elimination:
    tmp_4 = input_5 + input_4
    tmp_5 = tmp_4.mean((2, 3), keepdim=False)
    tmp_7 = tmp_5  # identity operation
    tmp_8 = torch.nn.functional.batch_norm(tmp_7, in_0, in_1, in_3, in_2, False, 0.1, 1e-05)
    return (tmp_8, tmp_7)
    """
    # Addition of the two input tensors
    tmp_4 = input_5 + input_4
    # Mean reduction over spatial dimensions
    tmp_5 = tmp_4.mean((2, 3), keepdim=False)
    # Identity operation (this becomes observable output tmp_7)
    tmp_7 = tmp_5
    # Batch normalization with running statistics
    tmp_8 = torch.nn.functional.batch_norm(tmp_7, in_0, in_1, in_3, in_2, False, 0.1, 1e-05)
    
    # Return both outputs as required by the graph
    return tmp_8, tmp_7

def replacement_args(input_4, input_5, in_0, in_1, in_2, in_3):
    """Extract arguments for the fused operation"""
    return (input_4, input_5, in_0, in_1, in_2, in_3)

@triton.jit
def fused_add_mean_batchnorm_kernel(
    input4_ptr,
    input5_ptr,
    running_mean_ptr,
    running_var_ptr,
    weight_ptr,
    bias_ptr,
    output8_ptr,
    output7_ptr,
    batch_size,
    channels,
    height,
    width,
    momentum: tl.constexpr,
    eps: tl.constexpr,
    BLOCK_SIZE: tl.constexpr
):
    """
    Fused kernel that performs:
    - Addition: input4 + input5
    - Spatial mean computation
    - Batch normalization with running stats
    All optimized in a single kernel
    """
    # Program ID for this thread
    pid = tl.program_id(0)
    
    # Check if this thread is within bounds
    if pid >= batch_size:
        return
    
    # Accumulators for spatial sum
    spatial_sum = tl.zeros([channels], dtype=tl.float32)
    
    # Process all spatial elements for this batch
    for h in range(height):
        for w in range(width):
            # Load input4 and input5 values for this position
            input4_offset = (pid * channels * height * width) + (h * width * channels) + w
            input5_offset = input4_offset + (channels * height * width)
            
            input4_vals = tl.load(input4_ptr + input4_offset, mask=tl.arange(channels) < channels, other=0.0)
            input5_vals = tl.load(input5_ptr + input5_offset, mask=tl.arange(channels) < channels, other=0.0)
            
            # Add and accumulate for spatial mean
            summed_vals = input4_vals + input5_vals
            spatial_sum += summed_vals
    
    # Compute spatial mean
    spatial_mean = spatial_sum / (height * width)
    
    # Load batch norm parameters
    running_mean = tl.load(running_mean_ptr, mask=tl.arange(channels) < channels, other=0.0)
    running_var = tl.load(running_var_ptr, mask=tl.arange(channels) < channels, other=1.0) 
    weight = tl.load(weight_ptr, mask=tl.arange(channels) < channels, other=1.0)
    bias = tl.load(bias_ptr, mask=tl.arange(channels) < channels, other=0.0)
    
    # Apply batch normalization
    batch_norm_output = (spatial_mean - running_mean) * weight / tl.sqrt(running_var + eps) + bias
    
    # Store outputs
    output7_offset = pid * channels
    output8_offset = pid * channels
    
    for i in range(channels):
        tl.store(output7_ptr + output7_offset + i, spatial_mean[i])
        tl.store(output8_ptr + output8_offset + i, batch_norm_output[i])

@torch.fx.wrap
def fused_add_mean_batchnorm(input_4, input_5, in_0, in_1, in_2, in_3):
    """
    Fused function that performs addition, spatial mean, and batch normalization in one kernel
    """
    batch_size, channels, height, width = input_4.shape
    
    # Use optimal block size - one kernel per batch element
    BLOCK_SIZE = 1
    
    # Calculate grid dimensions - one program per batch element
    grid_size = batch_size
    
    # Create output tensors with same properties as input
    output7 = torch.empty((batch_size, channels), dtype=input_4.dtype, device=input_4.device)  # tmp_7 (intermediate mean)
    output8 = torch.empty((batch_size, channels), dtype=input_4.dtype, device=input_4.device)  # tmp_8 (batch norm result)
    
    # Launch the fused kernel
    fused_add_mean_batchnorm_kernel[grid_size](
        input4_ptr=input_4,
        input5_ptr=input_5,
        running_mean_ptr=in_0,      # in_0 is running_mean
        running_var_ptr=in_1,       # in_1 is running_var
        weight_ptr=in_3,            # in_3 is weight
        bias_ptr=in_2,              # in_2 is bias
        output8_ptr=output8,
        output7_ptr=output7,
        batch_size=batch_size,
        channels=channels,
        height=height,
        width=width,
        momentum=0.1,
        eps=1e-05,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return output8, output7

def replacement_func():
    """Return the fused add-mean-batch normalization function"""
    return fused_add_mean_batchnorm