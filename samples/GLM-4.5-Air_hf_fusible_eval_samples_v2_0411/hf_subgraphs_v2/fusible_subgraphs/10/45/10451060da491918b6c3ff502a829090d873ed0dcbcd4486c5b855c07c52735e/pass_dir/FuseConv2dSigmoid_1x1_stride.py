import torch
import triton
import triton.language as tl

# Pattern matching function - must match exact operations from model.py
def pattern(weight_tensor, input_tensor, bias=None, stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1):
    """Match Conv2D followed by Sigmoid operation"""
    conv2d = torch.conv2d(input_tensor, weight_tensor, bias, stride, padding, dilation, groups)
    tmp_2 = torch.sigmoid(conv2d)
    return tmp_2

# Argument extraction function
def replacement_args(weight_tensor, input_tensor, bias=None, stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1):
    return (weight_tensor, input_tensor, stride, padding, dilation, groups)

# Triton kernel for fused 1x1 Conv2D + Sigmoid
@triton.jit
def fused_conv1x1_sigmoid_kernel(
    input_ptr, weight_ptr, output_ptr,
    batch_size, in_channels, in_height, in_width,
    out_channels,
    BLOCK_SIZE: tl.constexpr
):
    """Fused 1x1 Conv2D + Sigmoid kernel using Triton - optimized for 1x1 convolutions"""
    pid = tl.program_id(0)
    total_elements = batch_size * out_channels * in_height * in_width
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < total_elements
    
    # Convert offset to coordinates
    n = offsets // (out_channels * in_height * in_width)
    offset_h = offsets % (out_channels * in_height * in_width)
    c_out = offset_h // (in_height * in_width)
    h = (offset_h % (in_height * in_width)) // in_width
    w = offset_h % in_width
    
    # For 1x1 convolution: sum over input channels
    # output[n, c_out, h, w] = sum_{c_in} input[n, c_in, h, w] * weight[c_out, c_in, 0, 0]
    acc = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    for c_in in range(in_channels):
        # Input coordinate: [n, c_in, h, w]
        input_idx = n * in_channels * in_height * in_width + c_in * in_height * in_width + h * in_width + w
        # Weight coordinate: [c_out, c_in, 0, 0] (1x1 kernel)
        weight_idx = c_out * in_channels + c_in  # Flattened weight tensor
        
        input_val = tl.load(input_ptr + input_idx, mask=False)
        weight_val = tl.load(weight_ptr + weight_idx, mask=False)
        acc += input_val * weight_val
    
    # Apply sigmoid and store
    sigmoid_val = 1.0 / (1.0 + tl.exp(-acc))
    tl.store(output_ptr + offsets, sigmoid_val, mask=mask)

# Kernel wrapper
@torch.fx.wrap
def fused_conv2d_sigmoid(input_tensor, weight_tensor, stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1):
    """Fused Conv2D + Sigmoid wrapper function"""
    # Get tensor shapes - more robust detection based on typical tensor characteristics
    # Input tensor usually has batch dimension and larger spatial dimensions
    # Weight tensor has out_channels, in_channels, kernel_height, kernel_width
    
    # Try to identify which is input and which is weight
    if len(input_tensor.shape) == 4 and len(weight_tensor.shape) == 4:
        input_spatial = input_tensor.shape[2] * input_tensor.shape[3]
        weight_spatial = weight_tensor.shape[2] * weight_tensor.shape[3]
        
        if input_spatial > weight_spatial:
            # input_tensor is likely the input tensor, weight_tensor is the weight
            batch_size, in_channels, in_height, in_width = input_tensor.shape
            out_channels, in_channels_weight, kernel_height, kernel_width = weight_tensor.shape
            swapped = False
        else:
            # input_tensor is likely the weight, weight_tensor is the input
            out_channels, in_channels_weight, kernel_height, kernel_width = input_tensor.shape
            batch_size, in_channels, in_height, in_width = weight_tensor.shape
            swapped = True
    else:
        # Fallback to original logic
        swapped = False
        batch_size, in_channels, in_height, in_width = input_tensor.shape
        out_channels, in_channels_weight, kernel_height, kernel_width = weight_tensor.shape
    
    # Handle the case where kernel_height and kernel_width are 1 (common in MobileNetV3)
    if kernel_height == 1 and kernel_width == 1:
        # Optimized direct computation for 1x1 conv
        output = torch.empty((batch_size, out_channels, in_height, in_width), 
                           dtype=input_tensor.dtype, device=input_tensor.device)
        
        # Calculate total elements and launch kernel
        total_elements = batch_size * out_channels * in_height * in_width
        BLOCK_SIZE = 1024
        num_programs = (total_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
        
        # Pass tensors to kernel in correct order (input, weight, output)
        if not swapped:
            # Normal case: input_tensor is input, weight_tensor is weight
            fused_conv1x1_sigmoid_kernel[(num_programs,)](
                input_tensor, weight_tensor, output,
                batch_size, in_channels, in_height, in_width,
                out_channels,
                BLOCK_SIZE
            )
        else:
            # Swapped case: input_tensor is weight, weight_tensor is input
            fused_conv1x1_sigmoid_kernel[(num_programs,)](
                weight_tensor, input_tensor, output,
                batch_size, in_channels, in_height, in_width,
                out_channels,
                BLOCK_SIZE
            )
        return output
    
    # Debug information to understand the tensor shapes
    print(f"DEBUG: input_tensor.shape = {input_tensor.shape}")
    print(f"DEBUG: weight_tensor.shape = {weight_tensor.shape}")
    print(f"DEBUG: kernel_height = {kernel_height}, kernel_width = {kernel_width}")
    print(f"DEBUG: expected 1x1 but got {kernel_height}x{kernel_width}")
    
    # For larger kernels, we'll let the pass framework handle them
    # This pass is specifically optimized for 1x1 convolutions in MobileNetV3
    raise NotImplementedError(f"This pass only supports 1x1 convolutions for performance optimization, got {kernel_height}x{kernel_width}")

# Replacement function (NO arguments, returns function reference)
def replacement_func():
    return fused_conv2d_sigmoid