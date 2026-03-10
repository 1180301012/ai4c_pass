import torch
import triton
import triton.language as tl

def pattern(input_tensor, weight, bias, scaling_factor, residual):
    """
    Pattern to match the fused sequence:
    conv2d(input_tensor, weight, bias) -> dropout (eliminated) -> 
    scaling_factor.unsqueeze(-1).unsqueeze(-1) * conv_out -> residual + scaled_conv
    
    This matches: conv2d + broadcast scaling + residual addition
    """
    # Conv2D operation
    conv_out = torch.conv2d(input_tensor, weight, bias, (1, 1), (0, 0), (1, 1), 1)
    
    # Broadcast scaling (unsqueeze + unsqueeze + multiplication)
    scaled_conv = scaling_factor.unsqueeze(-1).unsqueeze(-1) * conv_out
    
    # Residual addition
    out = residual + scaled_conv
    
    return out

def replacement_args(input_tensor, weight, bias, scaling_factor, residual):
    """Return arguments needed for replacement"""
    return (input_tensor, weight, bias, scaling_factor, residual)

@triton.jit
def fused_conv_scale_add_kernel(
    input_ptr,      # [batch, in_channels, height, width]
    weight_ptr,     # [out_channels, in_channels, 1, 1]
    bias_ptr,       # [out_channels]
    scaling_ptr,    # [out_channels]
    residual_ptr,   # [batch, out_channels, height, width]
    output_ptr,     # [batch, out_channels, height, width]
    batch_size,
    in_channels,
    out_channels,
    height,
    width,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fused kernel that combines:
    - Conv2D with 1x1 kernel
    - Broadcast scaling
    - Residual addition
    """
    # Each program handles one output feature in one spatial location
    pid = tl.program_id(0)
    
    # Calculate output coordinates
    out_c = pid % out_channels
    spatial_pid = pid // out_channels
    
    # Calculate spatial coordinates
    w = spatial_pid % width
    h = (spatial_pid // width) % height
    b = spatial_pid // (width * height)
    
    # Calculate input pointers for this output location
    # For 1x1 conv with stride 1, output location (b, out_c, h, w)
    # comes from input location (b, :, h, w)
    
    total_elements = batch_size * height * width
    
    # Process all input channels for this output location
    for c_offset in range(0, in_channels, BLOCK_SIZE):
        block_size = min(BLOCK_SIZE, in_channels - c_offset)
        
        # Load input channels for this spatial location
        input_vals = tl.load(input_ptr + (b * in_channels * height * width + 
                                         (c_offset) * height * width + 
                                         h * width + w), 
                           c_offset, block_size)
        
        # Load corresponding weight values
        weight_vals = tl.load(weight_ptr + (out_c * in_channels + c_offset) * 1 * 1, 
                            c_offset, block_size)
        
        # Compute convolution sum
        conv_sum = tl.sum(input_vals * weight_vals)
        
        # Add bias if exists
        if bias_ptr is not None:
            bias_val = tl.load(bias_ptr + out_c)
            conv_sum += bias_val
        
        # Load scaling factor
        scaling_val = tl.load(scaling_ptr + out_c)
        
        # Apply scaling
        scaled_val = conv_sum * scaling_val
        
        # Load residual value
        residual_val = tl.load(residual_ptr + (b * out_channels * height * width + 
                                             out_c * height * width + 
                                             h * width + w))
        
        # Add residual
        final_val = scaled_val + residual_val
        
        # Store result
        tl.store(output_ptr + (b * out_channels * height * width + 
                              out_c * height * width + 
                              h * width + w), final_val)

@torch.fx.wrap
def fused_conv_scale_add(input_tensor, weight, bias, scaling_factor, residual):
    """
    Fused operation combining conv2d + broadcast scaling + residual addition
    This is much more efficient than running these operations separately
    """
    batch_size, in_channels, height, width = input_tensor.shape
    out_channels = weight.shape[0]
    
    # Prepare output tensor
    output = torch.empty((batch_size, out_channels, height, width), 
                        dtype=input_tensor.dtype, device=input_tensor.device)
    
    # Calculate grid size
    total_out_elements = batch_size * out_channels * height * width
    BLOCK_SIZE = 128  # Number of input channels to process per program
    grid_size = (total_out_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Launch kernel
    fused_conv_scale_add_kernel[grid_size](
        input_ptr=input_tensor,
        weight_ptr=weight,
        bias_ptr=bias,
        scaling_ptr=scaling_factor,
        residual_ptr=residual,
        output_ptr=output,
        batch_size=batch_size,
        in_channels=in_channels,
        out_channels=out_channels,
        height=height,
        width=width,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return output

def replacement_func():
    """Return the fused conv-scale-add function"""
    return fused_conv_scale_add