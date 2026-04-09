import torch
import triton
import triton.language as tl

def pattern(input_tensor, weight_tensor):
    """
    Pattern: Conv2D with stride=(1,1) and groups=384 followed by mean over spatial dimensions with keepdim=True
    This matches the computation structure found in stride=(1,1) target graphs with groups=384
    """
    conv2d_output = torch.conv2d(input_tensor, weight_tensor, None, (1, 1), (1, 1), (1, 1), 384)
    mean_output = conv2d_output.mean((2, 3), keepdim=True)
    return conv2d_output, mean_output

def pattern_groups256(input_tensor, weight_tensor):
    """
    Pattern: Conv2D with stride=(1,1) and groups=256 followed by mean over spatial dimensions with keepdim=True
    This matches the computation structure found in stride=(1,1) target graphs with groups=256
    """
    conv2d_output = torch.conv2d(input_tensor, weight_tensor, None, (1, 1), (1, 1), (1, 1), 256)
    mean_output = conv2d_output.mean((2, 3), keepdim=True)
    return conv2d_output, mean_output

def replacement_args(input_tensor, weight_tensor):
    """
    Extract arguments for the replacement function
    Returns input tensor, weight tensor, and derived parameters
    """
    return (input_tensor, weight_tensor)

@triton.jit
def fused_conv2d_mean_kernel(
    input_ptr, weight_ptr, 
    conv_output_ptr, mean_ptr,
    batch_size, in_channels, height, width,
    out_channels,
    stride_h, stride_w,
    pad_h, pad_w,
    kernel_h, kernel_w,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fused Conv2D + Mean kernel that computes both convolution output and spatial mean
    in a single pass, reducing memory bandwidth and improving performance
    """
    # Program ids for tiling
    batch_idx = tl.program_id(0)
    out_c = tl.program_id(1)
    
    # Conv2D parameters
    stride = (stride_h, stride_w)
    padding = (pad_h, pad_w)
    
    # Calculate output dimensions
    out_height = (height + 2 * pad_h - kernel_h) // stride_h + 1
    out_width = (width + 2 * pad_w - kernel_w) // stride_w + 1
    
    # Calculate flattened indices
    input_offset = batch_idx * in_channels * height * width
    weight_offset = out_c * in_channels * kernel_h * kernel_w
    conv_output_offset = batch_idx * out_channels * out_height * out_width
    mean_offset = batch_idx * out_channels
    
    # Load input and weight with bounds checking
    input_ptr += input_offset
    weight_ptr += weight_offset
    conv_output_ptr += conv_output_offset
    mean_ptr += mean_offset
    
    # Accumulators for mean computation
    mean_sum = 0.0
    
    # Conv2D computation with mean accumulation
    for c in range(in_channels):
        for kh in range(kernel_h):
            for kw in range(kernel_w):
                # Calculate weight offset
                weight_idx = c * kernel_h * kernel_w + kh * kernel_w + kw
                weight_val = tl.load(weight_ptr + weight_idx)
                
                # Conv2D computation
                for oh in range(out_height):
                    for ow in range(out_width):
                        # Calculate input indices with stride and padding
                        ih = oh * stride_h - pad_h + kh
                        iw = ow * stride_w - pad_w + kw
                        
                        if 0 <= ih < height and 0 <= iw < width:
                            # Calculate input index
                            input_idx = c * height * width + ih * width + iw
                            input_val = tl.load(input_ptr + input_idx)
                            
                            # Conv2D operation
                            conv_val = input_val * weight_val
                            
                            # Store conv output
                            conv_idx = out_c * out_height * out_width + oh * out_width + ow
                            tl.store(conv_output_ptr + conv_idx, conv_val)
                            
                            # Accumulate for mean
                            mean_sum += conv_val
                        else:
                            # Handle padding by contributing zero
                            conv_idx = out_c * out_height * out_width + oh * out_width + ow
                            tl.store(conv_output_ptr + conv_idx, 0.0)
                            
    # Compute mean and store
    num_elements = out_height * out_width
    mean_val = mean_sum / num_elements
    tl.store(mean_ptr, mean_val)

@torch.fx.wrap
def fused_conv2d_mean_stride1(input_tensor, weight_tensor):
    """
    Wrapper function that launches the fused Conv2D + Mean kernel for stride=(1,1)
    """
    # Get tensor dimensions
    batch_size, in_channels, height, width = input_tensor.shape
    out_channels, _, kernel_h, kernel_w = weight_tensor.shape
    
    # Stride parameters for stride=(1,1)
    stride_h, stride_w = 1, 1
    pad_h, pad_w = 1, 1
    
    # Calculate output dimensions
    out_height = (height + 2 * pad_h - kernel_h) // stride_h + 1
    out_width = (width + 2 * pad_w - kernel_w) // stride_w + 1
    
    # Create output tensors
    conv_output = torch.empty((batch_size, out_channels, out_height, out_width), 
                             dtype=input_tensor.dtype, device=input_tensor.device)
    mean_output = torch.empty((batch_size, out_channels), 
                             dtype=input_tensor.dtype, device=input_tensor.device)
    
    # Set up grid dimensions
    grid = (batch_size, out_channels)
    
    # Launch kernel
    fused_conv2d_mean_kernel[grid](
        input_tensor, weight_tensor,
        conv_output, mean_output,
        batch_size, in_channels, height, width,
        out_channels,
        stride_h, stride_w,
        pad_h, pad_w,
        kernel_h, kernel_w,
        BLOCK_SIZE=1024
    )
    
    return conv_output, mean_output

@torch.fx.wrap
def fused_conv2d_mean_stride2(input_tensor, weight_tensor):
    """
    Wrapper function that launches the fused Conv2D + Mean kernel for stride=(2,2)
    """
    # Get tensor dimensions
    batch_size, in_channels, height, width = input_tensor.shape
    out_channels, _, kernel_h, kernel_w = weight_tensor.shape
    
    # Stride parameters for stride=(2,2)
    stride_h, stride_w = 2, 2
    pad_h, pad_w = 1, 1
    
    # Calculate output dimensions
    out_height = (height + 2 * pad_h - kernel_h) // stride_h + 1
    out_width = (width + 2 * pad_w - kernel_w) // stride_w + 1
    
    # Create output tensors
    conv_output = torch.empty((batch_size, out_channels, out_height, out_width), 
                             dtype=input_tensor.dtype, device=input_tensor.device)
    mean_output = torch.empty((batch_size, out_channels), 
                             dtype=input_tensor.dtype, device=input_tensor.device)
    
    # Set up grid dimensions
    grid = (batch_size, out_channels)
    
    # Launch kernel
    fused_conv2d_mean_kernel[grid](
        input_tensor, weight_tensor,
        conv_output, mean_output,
        batch_size, in_channels, height, width,
        out_channels,
        stride_h, stride_w,
        pad_h, pad_w,
        kernel_h, kernel_w,
        BLOCK_SIZE=1024
    )
    
    return conv_output, mean_output

def replacement_func():
    """
    Returns the fused function implementation for stride=(1,1) case
    """
    return fused_conv2d_mean_stride1