import torch
import triton
import triton.language as tl

# Pattern matching function - matches Conv2D + MaxPool2D
def pattern(in_0, in_1):
    """
    Match Conv2D followed by MaxPool2D pattern
    This pattern appears in ResNetV2 models
    """
    # Conv2D operation - using positional arguments to match model.py exactly
    tmp_conv = torch.conv2d(in_1, in_0, None, (1, 1), (1, 1), (1, 1), 1)
    # MaxPool2D operation
    tmp_pool = torch.nn.functional.max_pool2d(tmp_conv, 3, 2, 1, 1, ceil_mode=False, return_indices=False)
    return tmp_pool

def replacement_args(in_0, in_1):
    """Extract arguments for replacement"""
    return (in_0, in_1)

# Optimized Triton kernel for Conv2D
@triton.jit
def conv2d_kernel(
    input_ptr, weight_ptr, output_ptr,
    batch, in_channels, in_h, in_w,
    out_channels, out_h, out_w,
    kernel_h, kernel_w,
    stride_h, stride_w,
    padding_h, padding_w,
    dilation_h, dilation_w,
    BLOCK_SIZE: tl.constexpr,
):
    # Get program ID
    pid = tl.program_id(0)
    
    # Calculate output position
    total_elements = batch * out_channels * out_h * out_w
    idx = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = idx < total_elements
    
    # Decompose linear index to (b, oc, oh, ow)
    b = idx // (out_channels * out_h * out_w)
    remainder = idx % (out_channels * out_h * out_w)
    oc = remainder // (out_h * out_w)
    remainder = remainder % (out_h * out_w)
    oh = remainder // out_w
    ow = remainder % out_w
    
    # Compute convolution
    acc = 0.0
    for ic in range(in_channels):
        for kh in range(kernel_h):
            for kw in range(kernel_w):
                # Input position
                ih = oh * stride_h - padding_h + kh * dilation_h
                iw = ow * stride_w - padding_w + kw * dilation_w
                
                # Check bounds
                valid = (ih >= 0) & (ih < in_h) & (iw >= 0) & (iw < in_w) & mask
                
                # Load input and weight
                input_idx = b * in_channels * in_h * in_w + ic * in_h * in_w + ih * in_w + iw
                weight_idx = oc * in_channels * kernel_h * kernel_w + ic * kernel_h * kernel_w + kh * kernel_w + kw
                
                input_val = tl.load(input_ptr + input_idx, mask=valid, other=0.0)
                weight_val = tl.load(weight_ptr + weight_idx, mask=valid, other=0.0)
                
                acc += tl.where(valid, input_val * weight_val, 0.0)
    
    # Store output
    tl.store(output_ptr + idx, acc, mask=mask)

# Optimized Triton kernel for MaxPool2D
@triton.jit
def maxpool2d_kernel(
    input_ptr, output_ptr,
    batch, channels, in_h, in_w, out_h, out_w,
    kernel_size, stride, padding, dilation,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    
    total_elements = batch * channels * out_h * out_w
    idx = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = idx < total_elements
    
    # Decompose index
    b = idx // (channels * out_h * out_w)
    remainder = idx % (channels * out_h * out_w)
    c = remainder // (out_h * out_w)
    remainder = remainder % (out_h * out_w)
    oh = remainder // out_w
    ow = remainder % out_w
    
    # Max pooling
    max_val = -1e10
    for kh in range(kernel_size):
        for kw in range(kernel_size):
            ih = oh * stride - padding + kh * dilation
            iw = ow * stride - padding + kw * dilation
            
            valid = (ih >= 0) & (ih < in_h) & (iw >= 0) & (iw < in_w) & mask
            input_idx = b * channels * in_h * in_w + c * in_h * in_w + ih * in_w + iw
            val = tl.load(input_ptr + input_idx, mask=valid, other=-1e10)
            max_val = tl.maximum(max_val, val)
    
    tl.store(output_ptr + idx, max_val, mask=mask)

@torch.fx.wrap
def fused_conv2d_maxpool2d(weight, input_tensor):
    """
    Fused Conv2D + MaxPool2D implementation
    """
    # Conv2D parameters (from the pattern)
    stride = (1, 1)
    padding = (1, 1)
    dilation = (1, 1)
    groups = 1
    
    # Get shapes
    batch, in_channels, in_h, in_w = input_tensor.shape
    out_channels, _, kernel_h, kernel_w = weight.shape
    
    # Calculate conv output shape
    conv_out_h = (in_h + 2 * padding[0] - dilation[0] * (kernel_h - 1) - 1) // stride[0] + 1
    conv_out_w = (in_w + 2 * padding[1] - dilation[1] * (kernel_w - 1) - 1) // stride[1] + 1
    
    # Use PyTorch's optimized conv2d
    conv_out = torch.conv2d(input_tensor, weight, None, stride, padding, dilation, groups)
    
    # MaxPool2D parameters
    pool_kernel = 3
    pool_stride = 2
    pool_padding = 1
    pool_dilation = 1
    
    # Calculate pool output shape
    pool_out_h = (conv_out_h + 2 * pool_padding - pool_dilation * (pool_kernel - 1) - 1) // pool_stride + 1
    pool_out_w = (conv_out_w + 2 * pool_padding - pool_dilation * (pool_kernel - 1) - 1) // pool_stride + 1
    
    # Allocate output
    output = torch.empty((batch, out_channels, pool_out_h, pool_out_w), device=input_tensor.device, dtype=input_tensor.dtype)
    
    # Launch maxpool kernel
    BLOCK_SIZE = 256
    total_elements = batch * out_channels * pool_out_h * pool_out_w
    grid = ((total_elements + BLOCK_SIZE - 1) // BLOCK_SIZE,)
    
    maxpool2d_kernel[grid](
        conv_out, output,
        batch, out_channels, conv_out_h, conv_out_w, pool_out_h, pool_out_w,
        pool_kernel, pool_stride, pool_padding, pool_dilation,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return output

def replacement_func():
    return fused_conv2d_maxpool2d