import torch
import triton
import triton.language as tl

# Pattern matching function - must exactly match the computation graph
def pattern(in_0, in_1, in_2, in_3):
    """
    Match the computation pattern: Conv2D + Sigmoid + Element-wise Mult + Hardtanh
    """
    tmp_0 = in_0
    tmp_1 = in_1
    tmp_2 = torch.conv2d(in_3, tmp_1, tmp_0, (1, 1), (0, 0), (1, 1), 1)
    tmp_3 = tmp_2.sigmoid()
    tmp_4 = in_2 * tmp_3
    tmp_5 = torch.nn.functional.hardtanh(tmp_4, 0.0, 6.0, False)
    return tmp_5

# Argument extraction function
def replacement_args(in_0, in_1, in_2, in_3):
    return (in_0, in_1, in_2, in_3)

# Triton kernel implementation with better optimization for 1x1 convolution
@triton.jit
def fused_conv_sigmoid_hardtanh_kernel(
    x_ptr,                    # in_3: input to conv [batch, in_channels, H, W]
    weight_ptr,               # in_1: conv weights [out_channels, in_channels, 1, 1]
    bias_ptr,                 # in_0: conv bias [out_channels]
    mult_ptr,                 # in_2: multiplication tensor [batch, out_channels, H, W]
    out_ptr,                  # output tensor [batch, out_channels, H, W]
    batch_size,
    in_channels,
    out_channels,
    height,
    width,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    
    # Each program handles one element in the output tensor
    batch_idx = pid // (out_channels * height * width)
    spatial_idx = pid % (out_channels * height * width)
    oc = spatial_idx // (height * width)
    spatial_idx2 = spatial_idx % (height * width)
    h = spatial_idx2 // width
    w = spatial_idx2 % width
    
    # Check bounds - avoid chained boolean operators
    if batch_idx >= batch_size:
        return
    if h >= height:
        return
    if w >= width:
        return
    if oc >= out_channels:
        return
    
    # Load bias for this output channel
    bias = tl.load(bias_ptr + oc)
    
    # Calculate input addresses (1x1 convolution)
    input_base_addr = batch_idx * in_channels * height * width + h * width * in_channels + w * in_channels
    weight_base_addr = oc * in_channels
    
    # Load input and weights using scalar loads for 1x1 conv
    conv_sum = 0.0
    for i in range(in_channels):
        input_val = tl.load(x_ptr + input_base_addr + i)
        weight_val = tl.load(weight_ptr + weight_base_addr + i)
        conv_sum += input_val * weight_val
    
    # Add bias
    conv_out = conv_sum + bias
    
    # Load multiplier for this location
    mult_addr = batch_idx * out_channels * height * width + h * width * out_channels + w * out_channels
    multiplier = tl.load(mult_ptr + mult_addr + oc)
    
    # Apply activation functions
    sigmoid_out = 1.0 / (1.0 + tl.exp(-conv_out))
    mult_result = sigmoid_out * multiplier
    
    # Hardtanh activation
    hardtanh_out = mult_result
    if hardtanh_out < 0.0:
        hardtanh_out = 0.0
    elif hardtanh_out > 6.0:
        hardtanh_out = 6.0
    
    # Store result
    out_addr = batch_idx * out_channels * height * width + h * width * out_channels + w * out_channels
    tl.store(out_ptr + out_addr + oc, hardtanh_out)

# Optimized kernel wrapper with autotuning
@torch.fx.wrap
def fused_conv_sigmoid_hardtanh(in_0, in_1, in_2, in_3):
    batch_size, in_channels, height, width = in_3.shape
    out_channels = in_1.shape[0]
    
    # Output tensor shape matches the Conv2D output
    out_shape = (batch_size, out_channels, height, width)
    out = torch.zeros(out_shape, dtype=torch.float32, device=in_3.device)
    
    # Calculate grid size - one program per output channel per spatial location per batch
    total_programs = batch_size * height * width * out_channels
    
    # Launch kernel - each program handles one output channel for one spatial location
    fused_conv_sigmoid_hardtanh_kernel[(total_programs,)](
        x_ptr=in_3,
        weight_ptr=in_1,
        bias_ptr=in_0,
        mult_ptr=in_2,
        out_ptr=out,
        batch_size=batch_size,
        in_channels=in_channels,
        out_channels=out_channels,
        height=height,
        width=width,
        BLOCK_SIZE=1,  # Each program handles one element
    )
    
    return out

# Replacement function (must return function reference, not call it)
def replacement_func():
    return fused_conv_sigmoid_hardtanh