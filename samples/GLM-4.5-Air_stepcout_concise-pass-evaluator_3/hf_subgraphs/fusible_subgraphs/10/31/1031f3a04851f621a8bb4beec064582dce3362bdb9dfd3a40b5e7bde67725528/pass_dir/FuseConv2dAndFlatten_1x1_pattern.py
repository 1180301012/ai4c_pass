import torch
import triton
import triton.language as tl

# Pattern matching function
def pattern(in_0, in_1, in_2):
    """Pattern: Conv2D with 1x1 kernel followed by flatten along spatial dimensions"""
    tmp_0 = in_0
    tmp_1 = in_1
    tmp_2 = torch.conv2d(in_2, tmp_1, tmp_0, (1, 1), (0, 0), (1, 1), 1)
    tmp_3 = torch.flatten(tmp_2, 2)
    return tmp_3

# Argument extraction function
def replacement_args(in_0, in_1, in_2):
    """Extract arguments for the fused convolution and flatten operation"""
    return (in_0, in_1, in_2)

# Optimized fused kernel
@triton.jit
def fused_conv2d_flatten_kernel(
    input_ptr,      # [B, C_in, H, W]
    weight_ptr,     # [C_out, C_in, 1, 1] 
    bias_ptr,       # [C_out]
    output_ptr,     # [B, C_out, H*W]
    batch_size: tl.constexpr,
    in_channels: tl.constexpr,
    out_channels: tl.constexpr,
    height: tl.constexpr,
    width: tl.constexpr,
    spatial_size: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Fused kernel: Conv2D (1x1) + flatten spatial dimensions"""
    pid = tl.program_id(0)
    
    # Each program handles one spatial position (x,y) across all batches and output channels
    spatial_idx = pid % spatial_size
    batch_idx = (pid // spatial_size) % batch_size
    out_ch_idx = pid // (spatial_size * batch_size)
    
    if out_ch_idx >= out_channels:
        return
    
    # Compute spatial coordinates
    y = spatial_idx // width
    x = spatial_idx % width
    
    # Accumulate convolution result
    result = 0.0
    # Use fixed-size loops with masks to handle dynamic bounds
    offset = tl.arange(0, BLOCK_SIZE)
    for ch_base in range(0, in_channels, BLOCK_SIZE):
        # Check if we're still within bounds
        in_bounds = ch_base + offset < in_channels
        
        # Load 1x1 weight segment for current output channel
        weights = tl.load(weight_ptr + out_ch_idx * in_channels + ch_base + offset, 
                         mask=in_bounds, other=0.0)
        
        # Load input segment for current batch position and channel
        inputs = tl.load(input_ptr + batch_idx * in_channels * height * width + 
                        (ch_base + offset) * height * width + y * width + x, 
                         mask=in_bounds, other=0.0)
        
        # Only accumulate valid elements
        valid_elements = tl.where(in_bounds, weights * inputs, 0.0)
        result += tl.sum(valid_elements)
    
    # Add bias
    bias = tl.load(bias_ptr + out_ch_idx)
    final_result = result + bias
    
    # Store result in flattened output
    output_idx = batch_idx * out_channels * spatial_size + out_ch_idx * spatial_size + spatial_idx
    tl.store(output_ptr + output_idx, final_result)

# Optimized fused kernel with better block size selection
@triton.jit
def fused_conv2d_flatten_kernel(
    input_ptr,      # [B, C_in, H, W]
    weight_ptr,     # [C_out, C_in, 1, 1] 
    bias_ptr,       # [C_out]
    output_ptr,     # [B, C_out, H*W]
    batch_size: tl.constexpr,
    in_channels: tl.constexpr,
    out_channels: tl.constexpr,
    height: tl.constexpr,
    width: tl.constexpr,
    spatial_size: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Fused kernel: Conv2D (1x1) + flatten spatial dimensions - optimized version"""
    pid = tl.program_id(0)
    
    # Each program handles one spatial position (x,y) across all batches and output channels
    spatial_idx = pid % spatial_size
    batch_idx = (pid // spatial_size) % batch_size
    out_ch_idx = pid // (spatial_size * batch_size)
    
    if out_ch_idx >= out_channels:
        return
    
    # Compute spatial coordinates
    y = spatial_idx // width
    x = spatial_idx % width
    
    # Accumulate convolution result
    result = 0.0
    # Use fixed-size loops with masks to handle dynamic bounds
    offset = tl.arange(0, BLOCK_SIZE)
    for ch_base in range(0, in_channels, BLOCK_SIZE):
        # Check if we're still within bounds
        in_bounds = ch_base + offset < in_channels
        
        # Load 1x1 weight segment for current output channel
        weights = tl.load(weight_ptr + out_ch_idx * in_channels + ch_base + offset, 
                         mask=in_bounds, other=0.0)
        
        # Load input segment for current batch position and channel
        inputs = tl.load(input_ptr + batch_idx * in_channels * height * width + 
                        (ch_base + offset) * height * width + y * width + x, 
                         mask=in_bounds, other=0.0)
        
        # Only accumulate valid elements
        result += tl.sum(weights * inputs)
    
    # Add bias
    bias = tl.load(bias_ptr + out_ch_idx)
    final_result = result + bias
    
    # Store result in flattened output
    output_idx = batch_idx * out_channels * spatial_size + out_ch_idx * spatial_size + spatial_idx
    tl.store(output_ptr + output_idx, final_result)

@torch.fx.wrap
def fused_conv2d_flatten_forward(bias, weight, input_tensor):
    """Wrapper function to launch the fused kernel with optimized block size selection"""
    B, C_in, H, W = input_tensor.shape
    C_out = weight.shape[0]
    spatial_size = H * W
    
    # Output shape: [B, C_out, H*W]
    output = torch.empty((B, C_out, spatial_size), 
                        dtype=input_tensor.dtype, 
                        device=input_tensor.device)
    
    # Optimized block size selection based on input dimensions
    if B * C_out > 1000:
        # Large workloads: use larger block sizes
        BLOCK_SIZE = 256
    elif B * C_out > 100:
        # Medium workloads: use medium block sizes  
        BLOCK_SIZE = 128
    else:
        # Small workloads: use smaller block sizes to avoid overhead
        BLOCK_SIZE = 64
    
    # Calculate grid size: need to cover all combinations of (batch, out_channel, spatial_pos)
    total_elements = B * C_out * spatial_size
    num_programs = (total_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Launch kernel with optimized configuration
    fused_conv2d_flatten_kernel[(num_programs,)](
        input_tensor,
        weight, 
        bias,
        output,
        B,
        C_in,
        C_out,
        H,
        W,
        spatial_size,
        BLOCK_SIZE
    )
    
    return output

# Replacement function
def replacement_func():
    return fused_conv2d_flatten_forward