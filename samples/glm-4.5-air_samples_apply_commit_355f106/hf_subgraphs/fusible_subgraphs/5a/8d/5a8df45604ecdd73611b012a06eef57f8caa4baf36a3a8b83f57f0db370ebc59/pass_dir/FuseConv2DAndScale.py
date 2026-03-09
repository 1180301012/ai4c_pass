import torch
import triton
import triton.language as tl

def pattern(conv_input, conv_weight, conv_bias, scale_weight, add_input):
    """
    Pattern: Conv2D + Dropout (p=0) + Scaling + Addition
    Dropout with p=0 is a no-op, so we fuse: Conv2D + Scaling + Addition
    """
    # 1x1 conv2d operation
    conv_out = torch.conv2d(conv_input, conv_weight, conv_bias, (1, 1), (0, 0), (1, 1), 1)
    
    # Create scaling factors: [C] -> [C, 1, 1] for broadcasting
    scaling_factors = scale_weight.unsqueeze(-1).unsqueeze(-1)
    
    # Apply scaling (dropout with p=0 is identity, so we skip it)
    scaled_out = scaling_factors * conv_out
    
    # Add to the input
    result = add_input + scaled_out
    
    return conv_out, scaled_out, result  # Return intermediate values for observability

def replacement_args(conv_input, conv_weight, conv_bias, scale_weight, add_input):
    return (conv_input, conv_weight, conv_bias, scale_weight, add_input)

@triton.jit
def fused_conv2d_scale_add_kernel(
    input_ptr, weight_ptr, bias_ptr, scale_ptr, add_ptr, out_ptr,
    batch_size, in_channels, out_channels, height, width,
    BLOCK_SIZE_m: tl.constexpr, BLOCK_SIZE_n: tl.constexpr, BLOCK_SIZE_k: tl.constexpr
):
    """
    Fused kernel for Conv2D (1x1) + Scaling + Addition
    Since it's a 1x1 conv, we can optimize it as a matrix multiplication
    """
    # Each program handles one output channel
    c = tl.program_id(0)
    
    # Load scale factor for this output channel
    scale_factor = tl.load(scale_ptr + c)
    
    # Compute the starting position in the output
    out_offset = c * batch_size * height * width
    add_offset = c * batch_size * height * width
    
    # Process spatial positions with within-channel parallelism
    for h in range(0, height, BLOCK_SIZE_m):
        for w in range(0, width, BLOCK_SIZE_n):
            # Load add input
            add_vals = tl.load(add_ptr + add_offset + h * width + w, mask=(h < height) & (w < width))
            
            # Initialize output with add input (since 1x1 conv is just linear transformation)
            out_vals = add_vals
            
            # Process each batch
            for b in range(batch_size):
                batch_offset = b * height * width
                spatial_offset = batch_offset + h * width + w
                
                # Load input features and apply weight
                for k in range(in_channels):
                    in_val = tl.load(input_ptr + spatial_offset + k * batch_size * height * width, 
                                   mask=(b < batch_size) & (h < height) & (w < width))
                    weight_val = tl.load(weight_ptr + c * in_channels + k)
                    out_vals += in_val * weight_val
            
            # Apply scaling and store
            out_vals = out_vals * scale_factor + add_vals
            tl.store(out_ptr + out_offset + h * width + w, out_vals, mask=(h < height) & (w < width))

@torch.fx.wrap
def fused_conv2d_scale_add(conv_input, conv_weight, conv_bias, scale_weight, add_input):
    """
    Fused implementation of Conv2D + Scaling + Addition
    """
    batch_size, in_channels, height, width = conv_input.shape
    out_channels = conv_weight.shape[0]
    
    # Create output tensor
    output = torch.empty((batch_size, out_channels, height, width), 
                        dtype=conv_input.dtype, device=conv_input.device)
    
    # Set up Triton kernel launch
    BLOCK_SIZE_m = 16  # spatial height tiles
    BLOCK_SIZE_n = 16  # spatial width tiles
    BLOCK_SIZE_k = 128  # reduction dimension
    
    # Calculate grid size
    grid = (out_channels,)
    
    # Launch kernel
    fused_conv2d_scale_add_kernel[grid](
        input_ptr=conv_input,
        weight_ptr=conv_weight,
        bias_ptr=conv_bias,
        scale_ptr=scale_weight,
        add_ptr=add_input,
        out_ptr=output,
        batch_size=batch_size,
        in_channels=in_channels,
        out_channels=out_channels,
        height=height,
        width=width,
        BLOCK_SIZE_m=BLOCK_SIZE_m,
        BLOCK_SIZE_n=BLOCK_SIZE_n,
        BLOCK_SIZE_k=BLOCK_SIZE_k,
    )
    
    return output

def replacement_func():
    return fused_conv2d_scale_add