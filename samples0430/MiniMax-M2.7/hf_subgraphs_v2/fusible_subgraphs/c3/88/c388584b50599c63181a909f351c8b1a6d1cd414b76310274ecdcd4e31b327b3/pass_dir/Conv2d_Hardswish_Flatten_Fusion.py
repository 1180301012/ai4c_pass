import torch
import triton
import triton.language as tl


# Autotune configurations for the fused kernel
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 64, 'num_warps': 4}, num_stages=2),
        triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 32, 'num_warps': 4}, num_stages=2),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 16, 'num_warps': 4}, num_stages=2),
        triton.Config({'BLOCK_SIZE_M': 8, 'BLOCK_SIZE_N': 128, 'num_warps': 4}, num_stages=2),
    ],
    key=['batch_size', 'out_channels'],
)
@triton.jit
def fused_conv2d_hardswish_kernel(
    input_ptr, weight_ptr, bias_ptr, output_ptr,
    batch_size, out_channels, in_channels,
    output_stride, input_stride, weight_stride,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, num_warps: tl.constexpr,
):
    """
    Fused Conv2d + Hardswish kernel with tiled computation.
    
    Each program handles a tile of size (BLOCK_SIZE_M x BLOCK_SIZE_N).
    - BLOCK_SIZE_M: number of output channels per thread block
    - BLOCK_SIZE_N: number of batch elements per thread block
    
    Each thread in a block computes part of the convolution for its tile.
    """
    # Program's starting position
    oc_start = tl.program_id(0) * BLOCK_SIZE_M
    bs_start = tl.program_id(1) * BLOCK_SIZE_N
    
    # Output and input offsets for this block
    offs_oc = oc_start + tl.arange(0, BLOCK_SIZE_M)
    offs_bs = bs_start + tl.arange(0, BLOCK_SIZE_N)
    
    # Create masks for valid elements
    oc_mask = offs_oc < out_channels
    bs_mask = offs_bs < batch_size
    
    # Initialize accumulator for convolution
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    # Reduction loop over input channels
    for ic in range(in_channels):
        # Load input elements: [batch, in_channels] 
        inp_offsets = offs_bs[:, None] * input_stride + ic * 1 + ic  # Simplified offset
        inp = tl.load(input_ptr + offs_bs[:, None] * 1 + offs_bs, mask=bs_mask, other=0.0)
        
        # Actually, let's compute offsets differently
        # input[b, ic] is at offset b * in_channels + ic
        inp_offsets = offs_bs[None, :] * in_channels + ic
        inp = tl.load(input_ptr + inp_offsets, mask=bs_mask, other=0.0)
        
        # Load weight elements: [out_channels, in_channels]
        # weight[oc, ic] is at offset oc * in_channels + ic
        weight_offsets = offs_oc[:, None] * in_channels + ic
        weight = tl.load(weight_ptr + weight_offsets, mask=oc_mask, other=0.0)
        
        # Multiply and accumulate
        acc += inp * weight
    
    # Add bias
    bias_offsets = offs_oc
    bias = tl.load(bias_ptr + bias_offsets, mask=oc_mask, other=0.0)
    conv_out = acc + bias[None, :]
    
    # Apply hardswish: x * relu6(x + 3) / 6
    x_plus_3 = conv_out + 3.0
    relu6 = tl.minimum(tl.maximum(x_plus_3, 0.0), 6.0)
    result = conv_out * relu6 / 6.0
    
    # Store result
    output_offsets = offs_bs[None, :] * output_stride + offs_oc[:, None]
    tl.store(output_ptr + output_offsets, result, mask=oc_mask[None, :] & bs_mask[:, None])


# Fallback kernel for small batch sizes or when 2D grid is not suitable
@triton.jit
def fused_conv_hardswish_fallback_kernel(
    input_ptr, weight_ptr, bias_ptr, output_ptr,
    batch_size, out_channels, in_channels,
):
    """
    Fallback kernel where each program computes one output element.
    Less efficient but works for all sizes.
    """
    oc = tl.program_id(0)
    bs = tl.program_id(1)
    
    # Output offset
    output_offset = bs * out_channels + oc
    
    # Compute convolution
    acc = 0.0
    for ic in range(in_channels):
        input_offset = bs * in_channels + ic
        weight_offset = oc * in_channels + ic
        inp = tl.load(input_ptr + input_offset)
        weight = tl.load(weight_ptr + weight_offset)
        acc += inp * weight
    
    # Add bias
    bias_val = tl.load(bias_ptr + oc)
    conv_out = acc + bias_val
    
    # Apply hardswish
    x_plus_3 = conv_out + 3.0
    relu6 = tl.minimum(tl.maximum(x_plus_3, 0.0), 6.0)
    result = conv_out * relu6 / 6.0
    
    # Store
    tl.store(output_ptr + output_offset, result)


@torch.fx.wrap
def fused_conv2d_hardswish(bias, weight, x):
    """
    Fused Conv2d + Hardswish + Reshape operation.
    
    Optimized for 1x1 convolutions on GPU using Triton.
    
    Args:
        bias: Bias tensor [out_channels] (in_0)
        weight: Weight tensor [out_channels, in_channels, 1, 1] (in_1)
        x: Input tensor [batch, in_channels, 1, 1] (in_2)
    
    Returns:
        Output tensor [batch, out_channels]
    """
    # Get tensor dimensions
    batch_size = x.shape[0]
    in_channels = x.shape[1]
    out_channels = weight.shape[0]
    
    # Allocate output
    output = torch.empty((batch_size, out_channels), dtype=x.dtype, device=x.device)
    
    # Grid dimensions
    # For large batch sizes, use a tiled kernel
    if batch_size >= 32 and out_channels >= 64:
        BLOCK_SIZE_M = 32
        BLOCK_SIZE_N = 64
        grid_m = (out_channels + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M
        grid_n = (batch_size + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N
        grid = (grid_m, grid_n)
        
        fused_conv2d_hardswish_kernel[grid](
            x, weight, bias, output,
            batch_size, out_channels, in_channels,
            out_channels, in_channels, in_channels,
        )
    else:
        # Use fallback kernel with 2D grid
        grid = (out_channels, batch_size)
        fused_conv_hardswish_fallback_kernel[grid](
            x, weight, bias, output,
            batch_size, out_channels, in_channels,
        )
    
    return output


def pattern(in_0, in_1, in_2):
    """
    Match: Conv2d -> Hardswish -> Flatten(1, -1)
    
    in_0: bias tensor [out_channels]
    in_1: weight tensor [out_channels, in_channels, 1, 1]
    in_2: input tensor [batch, in_channels, 1, 1]
    """
    conv2d = torch.conv2d(in_2, in_1, in_0, (1, 1), (0, 0), (1, 1), 1)
    tmp_3 = torch.nn.functional.hardswish(conv2d, True)
    tmp_4 = tmp_3.flatten(1, -1)
    return tmp_4


def replacement_args(in_0, in_1, in_2):
    """
    Extract arguments for the fused kernel.
    """
    return (in_0, in_1, in_2)


def replacement_func():
    """
    Return the fused Conv2d + Hardswish function.
    """
    return fused_conv2d_hardswish