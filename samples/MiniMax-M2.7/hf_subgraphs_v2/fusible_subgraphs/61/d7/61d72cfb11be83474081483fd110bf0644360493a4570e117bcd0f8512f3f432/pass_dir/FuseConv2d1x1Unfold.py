import torch
import triton
import triton.language as tl


@triton.jit
def triton_conv_kernel(
    input_ptr,
    weight_ptr,
    output_ptr,
    out_channels,
    in_channels,
    out_h,
    out_w,
):
    """
    Optimized 1x1 conv2d kernel.
    
    Grid: (out_h, out_w, out_channels)
    - Each program computes one output element
    """
    # Program position
    h = tl.program_id(0)
    w = tl.program_id(1)
    ch = tl.program_id(2)
    
    # Base offset for loading from input [batch, channels, height, width]
    base_offset = h * out_w * in_channels + w * in_channels
    
    # Compute matmul: sum over input_channels of inp * weight[ch]
    result = tl.zeros([], dtype=tl.float32)
    
    # Loop over input channels
    for c_in in range(in_channels):
        inp_offset = base_offset + c_in
        inp_val = tl.load(input_ptr + inp_offset)
        w_offset = ch * in_channels + c_in
        wgt_val = tl.load(weight_ptr + w_offset)
        result += inp_val * wgt_val
    
    # Calculate output offset in conv2d output [batch, out_channels, out_h, out_w]
    out_offset = ch * out_h * out_w + h * out_w + w
    
    tl.store(output_ptr + out_offset, result.to(tl.bfloat16))


@torch.fx.wrap
def triton_conv_unfold(weight, input_tensor):
    """
    Optimized conv2d kernel.
    
    Args:
        weight: Conv2d weight of shape [128, 256, 1, 1]
        input_tensor: Conv2d input of shape [1, 256, 32, 32]
    
    Returns the conv2d output [1, 128, 32, 32].
    """
    # Weight shape: [out_channels, in_channels, 1, 1]
    out_channels, in_channels, _, _ = weight.shape
    
    # Input shape: [batch, channels, height, width]
    batch, _, out_h, out_w = input_tensor.shape
    
    # Grid: (out_h, out_w, out_channels)
    grid = (out_h, out_w, out_channels)
    
    # Output: [1, 128, 32, 32] - conv2d format
    conv_out = torch.empty((1, out_channels, out_h, out_w), 
                           dtype=input_tensor.dtype, device=input_tensor.device)
    
    triton_conv_kernel[grid](
        input_tensor, weight, conv_out,
        out_channels,
        in_channels,
        out_h, out_w,
    )
    
    return conv_out


def pattern(in_0, in_1):
    """
    Match conv2d + unfold pattern.
    """
    conv2d = torch.conv2d(in_1, in_0, None, (1, 1), (0, 0), (1, 1), 1)
    return conv2d


def replacement_args(in_0, in_1):
    """Extract arguments for the fused kernel."""
    return (in_0, in_1)  # (weight, input)


def replacement_func():
    """Return the fused kernel function."""
    return triton_conv_unfold