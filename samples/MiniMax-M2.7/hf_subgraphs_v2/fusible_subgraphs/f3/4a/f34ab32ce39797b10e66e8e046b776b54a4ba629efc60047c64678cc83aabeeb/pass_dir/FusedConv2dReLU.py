import torch
import triton
import triton.language as tl


@triton.jit
def fused_conv2d_relu_kernel(
    input_ptr,
    weight_ptr,
    bias_ptr,
    output_ptr,
    in_channels,
    out_channels,
    height,
    width,
    kernel_h: tl.constexpr,
    kernel_w: tl.constexpr,
    stride_h: tl.constexpr,
    stride_w: tl.constexpr,
    pad_h: tl.constexpr,
    pad_w: tl.constexpr,
    dilation_h: tl.constexpr,
    dilation_w: tl.constexpr,
    groups: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fused Conv2D + ReLU kernel.
    Applies 1x1 convolution followed by ReLU activation.
    """
    # Get position
    pid = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_w = tl.program_id(2)
    
    # Calculate output position
    out_offset = (pid * height * height * width + pid_h * width + pid_w) * out_channels
    
    # Load bias
    bias = tl.load(bias_ptr + tl.arange(0, out_channels)).to(tl.float32)
    
    # Accumulate result
    result = tl.zeros((out_channels,), dtype=tl.float32)
    
    # For 1x1 conv, we can do a simple inner product
    # For each input channel
    for c_in in range(in_channels):
        # Load weight for this output channel, input channel
        weight_offset = c_in * out_channels
        w = tl.load(weight_ptr + weight_offset + tl.arange(0, out_channels)).to(tl.float32)
        
        # Load input
        in_offset = (pid * in_channels * height * width + c_in * height * width + pid_h * width + pid_w)
        x = tl.load(input_ptr + in_offset).to(tl.float32)
        
        # Multiply accumulate
        result = result + w * x
    
    # Add bias
    result = result + bias
    
    # Apply ReLU: max(0, x)
    result = tl.where(result > 0, result, 0.0)
    
    # Store result
    tl.store(output_ptr + out_offset + tl.arange(0, out_channels), result)


@torch.fx.wrap
def fused_conv2d_relu_wrapper(input, weight, bias, stride, padding, dilation, groups):
    """
    Wrapper function for the fused Conv2D + ReLU kernel.
    For 1x1 convolutions (stride=1, padding=0, dilation=1).
    """
    # Only handle 1x1 conv case for simplicity
    if stride != (1, 1) or padding != (0, 0) or dilation != (1, 1):
        # Fall back to PyTorch for non-1x1 conv
        out = torch.conv2d(input, weight, bias, stride, padding, dilation, groups)
        return torch.nn.functional.relu(out, inplace=True)
    
    # Get dimensions
    batch, in_channels, height, width = input.shape
    out_channels = weight.shape[0]
    
    # Output shape
    out_height = height
    out_width = width
    
    # Allocate output
    output = torch.empty(batch, out_channels, out_height, out_width, 
                        dtype=input.dtype, device=input.device)
    
    # Configure kernel
    BLOCK_SIZE = min(1024, triton.next_power_of_2(out_channels))
    grid = (batch, out_height, out_width)
    
    fused_conv2d_relu_kernel[grid](
        input,
        weight,
        bias,
        output,
        in_channels,
        out_channels,
        height,
        width,
        1, 1,  # kernel size
        stride[0], stride[1],
        padding[0], padding[1],
        dilation[0], dilation[1],
        groups,
        BLOCK_SIZE,
    )
    
    return output


@torch.fx.wrap
def relu_wrapper(x):
    """
    Optimized ReLU wrapper using Triton kernel.
    """
    # Allocate output
    output = torch.empty_like(x)
    
    # Launch a simple parallel kernel
    n_elements = x.numel()
    BLOCK_SIZE = 1024
    num_programs = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    relu_kernel[(num_programs,)](
        x,
        output,
        n_elements,
        BLOCK_SIZE,
    )
    
    return output


@triton.jit
def relu_kernel(
    x_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    # Get program ID
    pid = tl.program_id(0)
    
    # Calculate offsets
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load values
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    
    # Apply ReLU: max(0, x)
    out = tl.where(x > 0, x, 0.0)
    
    # Store result
    tl.store(out_ptr + offsets, out, mask=mask)


def pattern(x):
    """
    Match the pattern: ReLU(inplace=True)
    """
    return torch.nn.functional.relu(x, inplace=True)


def replacement_args(x):
    return (x,)


def replacement_func():
    return relu_wrapper