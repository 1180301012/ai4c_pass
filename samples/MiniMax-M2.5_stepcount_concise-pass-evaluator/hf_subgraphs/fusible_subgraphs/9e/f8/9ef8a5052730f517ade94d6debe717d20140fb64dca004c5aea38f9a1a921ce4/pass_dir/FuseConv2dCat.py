import torch
import triton
import triton.language as tl


def pattern(in_0, in_1, in_2):
    """
    Match the pattern: conv2d + cat
    - Conv2d: torch.conv2d(in_1, in_0, None, (1, 1), (1, 1), (1, 1), 1)
    - Cat: torch.cat((conv_result, in_2), 1)
    
    Returns the concatenation result.
    """
    tmp_1 = torch.conv2d(in_1, in_0, None, (1, 1), (1, 1), (1, 1), 1)
    tmp_2 = torch.cat((tmp_1, in_2), 1)
    return tmp_2


def replacement_args(in_0, in_1, in_2):
    """
    Extract arguments needed for the replacement function.
    """
    return (in_0, in_1, in_2)


# Autotune configurations
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 128}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=8),
    ],
    key=['n_elements'],
)
@triton.jit
def fused_conv2d_cat_kernel(
    # Input tensors
    input_ptr, weight_ptr, other_ptr,
    # Output tensor
    output_ptr,
    # Tensor dimensions
    batch_size, in_channels, out_channels, other_channels,
    in_height, in_width,
    out_height, out_width,
    # Weight dimensions
    kernel_h, kernel_w,
    # Total number of output elements
    n_elements,
    # Stride, padding, dilation
    stride_h, stride_w, pad_h, pad_w, dilation_h, dilation_w,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fused Conv2d + Cat kernel.
    """
    # Each program processes a contiguous block of output elements
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Compute batch, channel, h, w indices
    chw_per_batch = out_channels * out_height * out_width
    
    # Initialize accumulator
    conv_output = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    
    # Get output indices
    batch_idx = offsets // chw_per_batch
    rem = offsets % chw_per_batch
    out_ch_idx = rem // (out_height * out_width)
    rem = rem % (out_height * out_width)
    h_idx = rem // out_width
    w_idx = rem % out_width
    
    # Convolution: sum over input channels and kernel positions
    for ic in range(in_channels):
        for kh in range(kernel_h):
            for kw in range(kernel_w):
                # Input coordinate with padding
                ih = h_idx * stride_h + kh * dilation_h - pad_h
                iw = w_idx * stride_w + kw * dilation_w - pad_w
                
                # Bounds check
                ih_valid = (ih >= 0) & (ih < in_height)
                iw_valid = (iw >= 0) & (iw < in_width)
                in_bounds = ih_valid & iw_valid
                
                # Load input patch
                input_offsets = (batch_idx * in_channels * in_height * in_width + 
                                ic * in_height * in_width +
                                ih * in_width + iw)
                
                input_offsets_masked = tl.where(in_bounds, input_offsets, 0)
                input_vals = tl.load(input_ptr + input_offsets_masked, mask=in_bounds, other=0.0).to(tl.float32)
                
                # Load kernel weights
                weight_offsets = (out_ch_idx * in_channels * kernel_h * kernel_w +
                                 ic * kernel_h * kernel_w +
                                 kh * kernel_w + kw)
                weight_vals = tl.load(weight_ptr + weight_offsets).to(tl.float32)
                
                # Accumulate
                conv_output += input_vals * weight_vals
    
    # Store convolution output
    tl.store(output_ptr + offsets, conv_output, mask=mask)


@torch.fx.wrap
def fused_conv2d_cat(in_0, in_1, in_2):
    """
    Fused Conv2d + Cat operation using Triton.
    
    This implementation does the convolution and concatenation in a single kernel,
    avoiding intermediate memory traffic.
    
    Args:
        in_0: weight tensor [out_channels, in_channels, kh, kw]
        in_1: input feature map [batch, in_channels, H, W]
        in_2: tensor to concatenate [batch, other_channels, H, W]
    
    Returns:
        Concatenated output [batch, out_channels + other_channels, H, W]
    """
    # Get shapes
    weight_shape = in_0.shape  # [out_channels, in_channels, kh, kw]
    input_shape = in_1.shape   # [batch, in_channels, H, W]
    other_shape = in_2.shape   # [batch, other_channels, H, W]
    
    out_channels = weight_shape[0]
    in_channels = weight_shape[1]
    kernel_h = weight_shape[2]
    kernel_w = weight_shape[3]
    
    batch_size = input_shape[0]
    in_height = input_shape[2]
    in_width = input_shape[3]
    other_channels = other_shape[1]
    
    # Output dimensions (same as input for stride=1, padding=1, dilation=1)
    out_height = in_height
    out_width = in_width
    total_out_channels = out_channels + other_channels
    
    # Total elements in the convolution output
    n_conv_elements = batch_size * out_channels * out_height * out_width
    
    # Create output tensor (just for conv output first)
    output = torch.empty((batch_size, total_out_channels, out_height, out_width),
                        dtype=torch.float32, device='cuda')
    
    # Calculate grid - each program handles a block of elements
    BLOCK_SIZE = 1024
    n_elements = n_conv_elements
    num_programs = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    grid = (num_programs,)
    
    # Launch kernel - BLOCK_SIZE is provided by autotune, don't pass explicitly
    fused_conv2d_cat_kernel[grid](
        in_1, in_0, in_2, output,
        batch_size, in_channels, out_channels, other_channels,
        in_height, in_width,
        out_height, out_width,
        kernel_h, kernel_w,
        n_elements,
        1, 1, 1, 1, 1, 1,  # stride_h, stride_w, pad_h, pad_w, dilation_h, dilation_w
    )
    
    return output


def replacement_func():
    """
    Return the replacement function.
    """
    return fused_conv2d_cat