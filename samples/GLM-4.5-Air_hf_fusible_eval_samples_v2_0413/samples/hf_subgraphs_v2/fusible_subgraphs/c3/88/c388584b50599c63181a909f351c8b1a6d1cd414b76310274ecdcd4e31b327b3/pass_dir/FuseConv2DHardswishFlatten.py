import torch
import triton
import triton.language as tl

# Pattern matching function - exactly matches the computation graph
def pattern(in_0, in_1, in_2):
    """Match Conv2D + Hardswish + Flatten pattern"""
    # Conv2D with exact arguments from model.py
    conv2d = torch.conv2d(in_2, in_1, in_0, (1, 1), (0, 0), (1, 1), 1)
    # Hardswish with in-place=True
    tmp_3 = torch.nn.functional.hardswish(conv2d, True)
    # Flatten from dim 1 to end
    tmp_4 = tmp_3.flatten(1, -1)
    return tmp_4

# Argument extraction function
def replacement_args(in_0, in_1, in_2):
    """Extract arguments for replacement kernel"""
    return (in_0, in_1, in_2)

@triton.jit
def fused_conv_hardswish_flatten_kernel(
    input_ptr,
    weight_ptr,
    bias_ptr,
    output_ptr,
    dtype: tl.constexpr,
    batch_size: tl.constexpr,
    in_channels: tl.constexpr,
    out_channels: tl.constexpr,
    n_elements: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Optimized fused Conv2D (1x1) + Hardswish + Flatten kernel"""
    # Each program handles one element from the block
    pid = tl.program_id(0)
    offset = pid  # Each program handles one element
    
    # Check if this offset is within bounds
    if offset >= n_elements:
        return
    
    # Calculate batch and output channel for this offset
    batch_idx = offset // out_channels
    out_channel_idx = offset % out_channels
    
    # Calculate input offset: [batch_idx, in_channels] -> flattened
    input_offset = batch_idx * in_channels
    
    # Calculate weight offset: [out_channel_idx, in_channels] -> flattened  
    weight_offset = out_channel_idx * in_channels
    
    # Initialize with bias
    bias = tl.load(bias_ptr + out_channel_idx)
    result = bias
    
    # Compute weighted sum over input channels
    for c_in in range(in_channels):
        in_offset = input_offset + c_in
        w_offset = weight_offset + c_in
        
        # Load input with bounds checking
        if in_offset < batch_size * in_channels:
            x = tl.load(input_ptr + in_offset)
        else:
            x = tl.cast(0.0, dtype)
            
        # Load weight with bounds checking  
        if w_offset < out_channels * in_channels:
            weight = tl.load(weight_ptr + w_offset)
        else:
            weight = tl.cast(0.0, dtype)
        
        # Accumulate: output += input * weight
        result += x * weight
    
    # Apply hardswish: x * relu6(x + 3) / 6 using Triton ops
    add_val = result + 3.0
    relu6_val = tl.maximum(tl.minimum(add_val, 6.0), 0.0)
    hardswish_val = result * relu6_val * 0.16666666666666666  # / 6
    
    # Store result
    tl.store(output_ptr + offset, hardswish_val)

@torch.fx.wrap
def fused_conv_hardswish_flatten(bias, weight, input_tensor):
    """Wrapper for fused Conv2D + Hardswish + Flatten operation"""
    
    # Input shapes
    batch_size, in_channels, height, width = input_tensor.shape
    out_channels = bias.shape[0]
    
    # 1x1 convolution, so output spatial dimensions are same as input
    assert height == 1 and width == 1, "Only 1x1 convolution supported"
    
    # Total elements in flattened output
    n_elements = batch_size * out_channels
    
    # Create output tensor with correct shape [batch_size, out_channels]  
    output = torch.empty((batch_size, out_channels), dtype=input_tensor.dtype, device=input_tensor.device)
    
    # Map torch dtype to triton dtype
    if input_tensor.dtype == torch.bfloat16:
        triton_dtype = tl.bfloat16
    elif input_tensor.dtype == torch.float16:
        triton_dtype = tl.float16
    elif input_tensor.dtype == torch.float32:
        triton_dtype = tl.float32
    else:
        raise ValueError(f"Unsupported dtype: {input_tensor.dtype}")
    
    # Launch kernel with 1D grid: one program per output element
    fused_conv_hardswish_flatten_kernel[(n_elements,)](
        input_ptr=input_tensor,
        weight_ptr=weight,
        bias_ptr=bias,
        output_ptr=output,
        dtype=triton_dtype,
        batch_size=batch_size,
        in_channels=in_channels,
        out_channels=out_channels,
        n_elements=n_elements,
        BLOCK_SIZE=1,  # Each program handles one element
    )
    
    return output

# Replacement function
def replacement_func():
    """Return the fused kernel function"""
    return fused_conv_hardswish_flatten