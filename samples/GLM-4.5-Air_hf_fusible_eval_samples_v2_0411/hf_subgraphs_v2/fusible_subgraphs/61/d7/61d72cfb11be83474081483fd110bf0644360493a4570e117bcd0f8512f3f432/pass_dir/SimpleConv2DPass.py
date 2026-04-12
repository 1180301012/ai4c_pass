import torch
import triton
import triton.language as tl

def pattern(in_0, in_1):
    """Pattern matching for just the Conv2D operation"""
    # Match exactly the conv2d operation from float32 version
    tmp_0 = in_0
    tmp_1 = torch.conv2d(in_1, tmp_0, None, (1, 1), (0, 0), (1, 1), 1)
    return tmp_1

def replacement_args(in_0, in_1):
    """Extract arguments for replacement"""
    return (in_0, in_1)

@triton.jit
def simple_conv2d_kernel(
    x_ptr,                # Input tensor: [1, 256, 32, 32]
    weight_ptr,           # Weight tensor: [128, 256, 1, 1]
    output_ptr,           # Output tensor: [1, 128, 32, 32]
    batch_size: tl.constexpr,   # 1
    in_channels: tl.constexpr,  # 256
    out_channels: tl.constexpr, # 128
    in_height: tl.constexpr,    # 32
    in_width: tl.constexpr,     # 32
    block_size: tl.constexpr,
):
    """Simple 1x1 convolution kernel"""
    pid = tl.program_id(0)
    total_elements = batch_size * out_channels * in_height * in_width
    block_start = pid * block_size
    block_end = min(block_start + block_size, total_elements)
    
    for idx in range(block_start, block_end):
        # Calculate coordinates: [B, C_out, H, W]
        out_channel = idx // (in_height * in_width) % out_channels
        height = idx // in_width % in_height
        width = idx % in_width
        
        # Perform 1x1 convolution: equivalent to matrix multiplication element-wise
        result = 0.0
        channels_per_group = in_channels // 64  # For better parallelization
        
        # Load input and weight for this output position
        input_offset = height * in_width
        weight_offset = out_channel * in_channels
        
        # Load input and weight with stride for better memory access
        input_val = tl.load(x_ptr + input_offset + height * in_width + width, mask=True, other=0.0)
        weight_val = tl.load(weight_ptr + weight_offset, mask=True, other=0.0)
        
        # Simple 1x1 convolution: element-wise multiplication 
        result = input_val * weight_val
        
        # Store result
        output_offset = out_channel * (in_height * in_width) + height * in_width + width
        tl.store(output_ptr + output_offset, result)

@torch.fx.wrap  
def simple_conv2d_fusion(weight_tensor, input_tensor):
    """Simple Conv2D operation using Triton"""
    batch, in_channels, in_height, in_width = input_tensor.shape
    out_channels, _, _, _ = weight_tensor.shape
    
    # Create output tensor
    output = torch.empty((batch, out_channels, in_height, in_width),
                        dtype=input_tensor.dtype,
                        device=input_tensor.device)
    
    # Launch kernel
    block_size = 1024
    total_elements = batch * out_channels * in_height * in_width
    num_blocks = (total_elements + block_size - 1) // block_size
    
    simple_conv2d_kernel[(num_blocks,)](
        x_ptr=input_tensor,
        weight_ptr=weight_tensor,
        output_ptr=output,
        batch_size=batch,
        in_channels=in_channels,
        out_channels=out_channels,
        in_height=in_height,
        in_width=in_width,
        block_size=block_size
    )
    
    return output

def replacement_func():
    """Return the simple Conv2D function"""
    return simple_conv2d_fusion