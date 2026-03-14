import torch
import triton
import triton.language as tl

# Pattern matching function
def pattern(in_0, in_1, in_2):
    """
    Conv2D + Permute + Reshape + Sigmoid fusion pattern
    
    This pattern matches the sequence:
    1. conv2d with stride=(1,1), padding=(0,0), dilation=(1,1), groups=1
    2. permute(0, 2, 3, 1) to change layout from [B,C,H,W] to [B,H,W,C]
    3. reshape to [B, H*W, num_channels]
    4. sigmoid activation
    """
    tmp_2 = torch.conv2d(in_2, in_1, in_0, (1, 1), (0, 0), (1, 1), 1)
    tmp_3 = tmp_2.permute(0, 2, 3, 1)
    tmp_4 = tmp_3.reshape(tmp_3.shape[0], -1, tmp_3.shape[-1])
    tmp_5 = torch.nn.functional.sigmoid(tmp_4)
    return tmp_5

# Argument extraction function
def replacement_args(in_0, in_1, in_2):
    return (in_0, in_1, in_2)

# Optimized Triton kernel
@triton.jit
def fused_conv_permute_sigmoid_kernel(
    input_ptr,  # [B, C_in, H, W]
    weight_ptr,  # [C_out, C_in, 1, 1]
    bias_ptr,    # [C_out]
    output_ptr,  # [B, H*W, C_out]
    batch_size,
    in_channels,
    out_channels,
    height,
    width,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles one block of the output
    pid = tl.program_id(0)
    block_size = BLOCK_SIZE
    
    # Calculate which elements this program handles
    n_elements = batch_size * height * width * out_channels
    offset = pid * block_size
    mask = offset < n_elements
    
    if not mask:
        return
    
    # Calculate positions for the current element
    idx = offset + tl.arange(0, block_size)
    batch_idx = idx // (height * width * out_channels)
    batch_idx = tl.minimum(batch_idx, batch_size - 1)
    
    spatial_idx = idx % (height * width * out_channels)
    channel_idx = spatial_idx // (height * width)
    spatial_idx = spatial_idx % (height * width)
    
    y_idx = spatial_idx // width
    x_idx = spatial_idx % width
    
    # Load bias
    bias = tl.load(bias_ptr + channel_idx, mask=True)
    
    # Calculate input position for convolution
    in_y = y_idx
    in_x = x_idx
    
    # Load input value for current position
    input_offset = batch_idx * in_channels * height * width + in_y * width + in_x
    input_val = tl.load(input_ptr + input_offset, mask=True)
    
    # Compute convolution with pointwise kernel
    conv_result = input_val * weight_ptr[channel_idx * in_channels] + bias
    
    # Apply sigmoid
    result = 1.0 / (1.0 + tl.exp(-conv_result))
    
    # Store result in output layout [B, H*W, C_out]
    output_offset = batch_idx * (height * width * out_channels) + spatial_idx * out_channels + channel_idx
    tl.store(output_ptr + output_offset, result, mask=mask)

@torch.fx.wrap  
def fused_conv_permute_sigmoid(input, weight, bias):
    # Get tensor shapes
    batch_size, in_channels, height, width = input.shape
    out_channels = bias.shape[0]
    
    # Calculate output shape [B, H*W, C_out]
    HxW = height * width
    total_elements = batch_size * HxW * out_channels
    
    # Kernel launch configuration
    BLOCK_SIZE = 1024
    num_programs = (total_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Create output tensor
    output = torch.empty((batch_size, HxW, out_channels), 
                        dtype=input.dtype, device=input.device)
    
    # Launch Triton kernel
    fused_conv_permute_sigmoid_kernel[(num_programs,)](
        input_ptr=input,
        weight_ptr=weight,
        bias_ptr=bias,
        output_ptr=output,
        batch_size=batch_size,
        in_channels=in_channels,
        out_channels=out_channels,
        height=height,
        width=width,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return output

# Replacement function (MUST be zero-argument function that returns a function reference)
def replacement_func():
    return fused_conv_permute_sigmoid