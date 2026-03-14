import torch
import triton
import triton.language as tl

def pattern(in_4, in_0, in_1, in_3, in_2):
    """
    Pattern for batch normalization operation
    Matches exactly the computation from model.py:
    tmp_5 = torch.nn.functional.batch_norm(in_4, in_0, in_1, in_3, in_2, False, 0.1, 0.001)
    """
    return torch.nn.functional.batch_norm(in_4, in_0, in_1, in_3, in_2, False, 0.1, 0.001)

def replacement_args(in_4, in_0, in_1, in_3, in_2):
    return (in_4, in_0, in_1, in_3, in_2)

@triton.jit
def optimized_batch_norm_kernel(
    input_ptr,
    running_mean_ptr,
    running_var_ptr,
    weight_ptr,
    bias_ptr,
    output_ptr,
    batch_size,
    num_features,
    height,
    width,
    eps: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_H: tl.constexpr,
):
    """
    Optimized Triton kernel for batch normalization
    Each program processes one feature channel and processes spatial blocks
    """
    # Program ID for the feature channel
    pid_c = tl.program_id(0)
    pid_h = tl.program_id(1)  # spatial height block
    
    # Get batch normalization parameters for this channel
    mean = tl.load(running_mean_ptr + pid_c)
    var = tl.load(running_var_ptr + pid_c)
    weight_val = tl.load(weight_ptr + pid_c)
    bias_val = tl.load(bias_ptr + pid_c)
    
    # Compute inverse standard deviation (sqrt(var + eps))
    inv_std = 1.0 / tl.sqrt(var + eps)
    
    # Calculate spatial offsets within this block
    h_offset = pid_h * BLOCK_SIZE_H + tl.arange(0, BLOCK_SIZE_H)
    w_offset = tl.arange(0, BLOCK_SIZE_N)
    
    # Create coordinate offsets
    h_coords = h_offset[:, None]
    w_coords = w_offset[None, :]
    
    # Calculate mask for valid spatial coordinates
    h_mask = h_coords < height
    w_mask = w_coords < width
    spatial_mask = h_mask & w_mask
    
    # Process each batch
    for b in range(batch_size):
        # Calculate memory addresses for this batch
        input_base = b * num_features * height * width + pid_c * height * width
        output_base = b * num_features * height * width + pid_c * height * width
        
        input_addr = input_ptr + input_base + h_coords * width + w_coords
        output_addr = output_ptr + output_base + h_coords * width + w_coords
        
        # Load input data with spatial mask
        x = tl.load(input_addr, mask=spatial_mask, other=0.0)
        
        # Apply batch normalization: y = (x - mean) * inv_std * weight + bias
        # This can be optimized to: y = x * (inv_std * weight) + (bias - mean * inv_std * weight)
        x_normalized = (x - mean) * inv_std
        y = x_normalized * weight_val + bias_val
        
        # Store result with spatial mask
        tl.store(output_addr, y, mask=spatial_mask)

@torch.fx.wrap  
def optimized_batch_norm(input, running_mean, running_var, weight, bias):
    """
    Optimized batch normalization implementation using Triton
    """
    batch_size, num_features, height, width = input.shape
    
    # Create output tensor
    output = torch.empty_like(input)
    
    # Set optimal block sizes - these can be autotuned for specific GPU
    BLOCK_SIZE_N = 32  # width block size
    BLOCK_SIZE_H = 32  # height block size
    
    # Calculate grid dimensions
    grid_c = num_features
    grid_h = (height + BLOCK_SIZE_H - 1) // BLOCK_SIZE_H
    
    # Launch kernel
    optimized_batch_norm_kernel[(grid_c, grid_h, 1)](
        input_ptr=input,
        running_mean_ptr=running_mean,
        running_var_ptr=running_var,
        weight_ptr=weight,
        bias_ptr=bias,
        output_ptr=output,
        batch_size=batch_size,
        num_features=num_features,
        height=height,
        width=width,
        eps=0.1,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_H=BLOCK_SIZE_H,
    )
    
    return output

def replacement_func():
    return optimized_batch_norm