import torch
import triton
import triton.language as tl

# Pattern matching function for the conv2d stream
def pattern(bias, weight, input_feat):
    # Match the conv2d operation exactly as in the model
    # bias: [out_channels] - 1D tensor
    # weight: [out_channels, in_channels, 1, 1] - 4D tensor  
    # input_feat: [batch, in_channels, height, width] - 4D tensor
    tmp_2 = torch.conv2d(input_feat, weight, bias, (1, 1), (0, 0), (1, 1), 1)
    return tmp_2

# Argument extraction function
def replacement_args(in_0, in_1, in_2):
    return (in_0, in_1, in_2)  # bias, weight, input_feat

# Triton kernel for optimized 1x1 conv2d
@triton.jit
def conv2d_1x1_kernel(
    input_ptr,      # [N, C, H, W] - input feature map
    weight_ptr,     # [OC, C, 1, 1] - weights
    bias_ptr,       # [OC] - bias
    output_ptr,     # [N, OC, H, W] - output
    n_batch: tl.constexpr,      # batch size
    n_in_channels: tl.constexpr,  # input channels
    n_out_channels: tl.constexpr, # output channels  
    height: tl.constexpr,     # feature map height
    width: tl.constexpr,      # feature map width
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles one output element
    pid = tl.program_id(0)
    n_elems = n_batch * n_out_channels * height * width
    
    if pid >= n_elems:
        return
        
    # Calculate indices
    batch = pid // (n_out_channels * height * width)
    out_channel = (pid // (height * width)) % n_out_channels
    h = (pid // width) % height  
    w = pid % width
    
    # Compute base pointers
    input_baseptr = input_ptr + batch * n_in_channels * height * width
    weight_baseptr = weight_ptr + out_channel * n_in_channels
    output_baseptr = output_ptr + pid
    
    # Load bias
    bias = tl.load(bias_ptr + out_channel)
    
    # Perform 1x1 convolution (element-wise multiplication and sum)
    result = bias
    for c in range(n_in_channels):
        input_val = tl.load(input_baseptr + c * height * width + h * width + w)
        weight_val = tl.load(weight_baseptr + c)
        result += input_val * weight_val
    
    # Store result
    tl.store(output_baseptr, result)

# Optimized conv2d wrapper
@torch.fx.wrap  
def optimized_conv2d_1x1(bias, weight, input_feat):
    # Get tensor shapes
    # bias: [out_channels] 
    # weight: [out_channels, in_channels, 1, 1]
    # input_feat: [batch, in_channels, height, width]
    
    n_out_channels = bias.shape[0]
    n_in_channels = weight.shape[1]
    n_batch, _, height, width = input_feat.shape
    
    # Create output tensor
    output = torch.empty((n_batch, n_out_channels, height, width), 
                        dtype=input_feat.dtype, device=input_feat.device)
    
    # Set block size and launch grid  
    BLOCK_SIZE = 128
    n_output_elems = n_batch * n_out_channels * height * width
    n_programs = (n_output_elems + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    conv2d_1x1_kernel[(n_programs,)](
        input_ptr=input_feat,
        weight_ptr=weight, 
        bias_ptr=bias,
        output_ptr=output,
        n_batch=n_batch,
        n_in_channels=n_in_channels,
        n_out_channels=n_out_channels,
        height=height,
        width=width,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return output

# Replacement function (returns function reference)
def replacement_func():
    return optimized_conv2d_1x1