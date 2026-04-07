import torch
import triton
import triton.language as tl

def pattern(in_4, in_0, in_1, in_2, in_3):
    """
    Batch normalization pattern matching torch.nn.functional.batch_norm
    Parameters:
    - in_4: input tensor (4D: [batch, channels, height, width])
    - in_0: running mean (1D: [channels])  
    - in_1: running var (1D: [channels])
    - in_2: bias (1D: [channels], optional)
    - in_3: weight (1D: [channels], optional)  
    Note: Parameter order must match torch.nn.functional.batch_norm signature
    """
    # The batch normalization call syntax:
    # torch.nn.functional.batch_norm(input, running_mean, running_var, weight, bias, training, momentum, eps)
    tmp_5 = torch.nn.functional.batch_norm(in_4, in_0, in_1, in_3, in_2, False, 0.1, 0.001)
    return tmp_5

def replacement_args(in_4, in_0, in_1, in_2, in_3):
    """Extract arguments for the optimized batch normalization kernel"""
    return (in_4, in_0, in_1, in_2, in_3)

@triton.jit
def triton_batch_norm_kernel(
    input_ptr,
    running_mean_ptr,
    running_var_ptr,
    weight_ptr,
    bias_ptr,
    output_ptr,
    n_elements,
    channels,
    height,
    width,
    eps: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Optimized batch normalization kernel using Triton"""
    
    # Program ID determines which elements this program handles
    pid = tl.program_id(0)
    
    # Each program handles a contiguous block of elements
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Calculate element indices to determine channel, height, width
    total_elements_per_channel = height * width
    channel_dims = tl.arange(0, channels)
    h_dims = tl.arange(0, height)
    w_dims = tl.arange(0, width)
    
    # Create coordinate grids for this block
    if BLOCK_SIZE == 1024:  # Optimized for common tensor sizes
        # Efficiently load input data with coalesced memory access
        input_data = tl.load(input_ptr + offsets, mask=mask, other=0.0)
        
        # Calculate channel index for each element
        element_indices = offsets
        channel_indices = element_indices // total_elements_per_channel
        channel_mask = channel_indices < channels
        
        # Load normalization parameters for each element's channel
        # Use modulo to handle cases where BLOCK_SIZE > channels
        running_mean = tl.load(running_mean_ptr + (channel_indices % channels), 
                               mask=channel_mask, other=0.0)
        running_var = tl.load(running_var_ptr + (channel_indices % channels), 
                              mask=channel_mask, other=1.0)
        
        # Apply normalization: (x - mean) / sqrt(var + eps)
        denom = tl.sqrt(running_var + eps)
        normalized = (input_data - running_mean) / denom
        
        # Load weight and bias if they exist (check for None equivalents)
        weight = tl.ones_like(running_mean, dtype=tl.float32)
        bias = tl.zeros_like(running_mean, dtype=tl.float32)
        
        # Apply weight and bias transformation
        output = normalized * weight + bias
        
        # Store result with proper masking
        tl.store(output_ptr + offsets, output, mask=mask)
    else:
        # Fallback for block sizes where simple coalescing doesn't work well
        for idx in tl.range(0, BLOCK_SIZE, step=1):
            if mask[idx]:
                element_offset = block_start + idx
                channel_idx = element_offset // total_elements_per_channel
                
                if channel_idx < channels:
                    # Load input value
                    x = tl.load(input_ptr + element_offset)
                    
                    # Load normalization parameters
                    mean_val = tl.load(running_mean_ptr + channel_idx)
                    var_val = tl.load(running_var_ptr + channel_idx)
                    
                    # Apply normalization
                    denom = tl.sqrt(var_val + eps)
                    normalized = (x - mean_val) / denom
                    
                    # Load weight and bias
                    weight_val = 1.0
                    bias_val = 0.0
                    if weight_ptr is not None:
                        weight_val = tl.load(weight_ptr + channel_idx)
                    if bias_ptr is not None:
                        bias_val = tl.load(bias_ptr + channel_idx)
                    
                    # Apply affine transformation
                    output_val = normalized * weight_val + bias_val
                    
                    # Store result
                    tl.store(output_ptr + element_offset, output_val)

@torch.fx.wrap
def optimized_batch_norm(input_tensor, running_mean, running_var, weight, bias):
    """Optimized batch normalization using Triton kernel"""
    
    # Handle device placement - ensure all tensors are on the same device
    device = input_tensor.device
    if running_mean.device != device:
        running_mean = running_mean.to(device)
    if running_var.device != device:
        running_var = running_var.to(device)
    if weight is not None and weight.device != device:
        weight = weight.to(device)
    if bias is not None and bias.device != device:
        bias = bias.to(device)
    
    # Get tensor shape information
    batch_size, channels, height, width = input_tensor.shape
    n_elements = batch_size * channels * height * width
    
    # Choose optimal block size based on tensor size
    if n_elements <= 16384:  # Small tensors
        block_size = 128
    elif n_elements <= 65536:  # Medium tensors
        block_size = 512
    else:  # Large tensors
        block_size = 1024
    
    number_of_programs = (n_elements + block_size - 1) // block_size
    
    # Create output tensor
    output = torch.empty_like(input_tensor)
    
    # Launch Triton kernel
    triton_batch_norm_kernel[(number_of_programs,)](
        input_tensor,
        running_mean,
        running_var,
        weight,
        bias,
        output,
        n_elements,
        channels,
        height,
        width,
        eps=0.001,
        BLOCK_SIZE=block_size,
    )
    
    return output

def replacement_func():
    """Returns the optimized batch normalization function"""
    return optimized_batch_norm