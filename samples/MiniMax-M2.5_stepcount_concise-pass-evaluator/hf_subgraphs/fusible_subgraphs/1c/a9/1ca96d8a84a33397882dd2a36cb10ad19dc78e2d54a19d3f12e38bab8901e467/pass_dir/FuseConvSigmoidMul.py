import torch
import triton
import triton.language as tl


# Pattern matching: Conv2D + Sigmoid + Element-wise Multiply
def pattern(in_0, in_1, in_5, in_6):
    """
    Match Conv2D + Sigmoid + Multiply pattern.
    in_0: bias [40]
    in_1: weight [40, 10, 1, 1]
    in_5: input to multiply [batch, 40, 32, 24]
    in_6: conv input [batch, 10, 1, 1]
    """
    # Conv2D with bias
    tmp_2 = torch.conv2d(in_6, in_1, in_0, (1, 1), (0, 0), (1, 1), 1)
    # Sigmoid activation
    tmp_3 = torch.sigmoid(tmp_2)
    # Element-wise multiply with broadcasting
    tmp_4 = in_5 * tmp_3
    return tmp_4


def replacement_args(in_0, in_1, in_5, in_6):
    return (in_0, in_1, in_5, in_6)


# Kernel with better memory access patterns
@triton.autotune(
    configs=[
        # Small tensors
        triton.Config({'BLOCK_SIZE_M': 1, 'BLOCK_SIZE_N': 256}, num_stages=2, num_warps=1),
        triton.Config({'BLOCK_SIZE_M': 1, 'BLOCK_SIZE_N': 512}, num_stages=2, num_warps=1),
        triton.Config({'BLOCK_SIZE_M': 1, 'BLOCK_SIZE_N': 1024}, num_stages=2, num_warps=2),
        # Medium tensors
        triton.Config({'BLOCK_SIZE_M': 1, 'BLOCK_SIZE_N': 512}, num_stages=3, num_warps=2),
        triton.Config({'BLOCK_SIZE_M': 1, 'BLOCK_SIZE_N': 1024}, num_stages=3, num_warps=2),
        triton.Config({'BLOCK_SIZE_M': 1, 'BLOCK_SIZE_N': 2048}, num_stages=3, num_warps=2),
        # Large tensors
        triton.Config({'BLOCK_SIZE_M': 1, 'BLOCK_SIZE_N': 1024}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 1, 'BLOCK_SIZE_N': 2048}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 1, 'BLOCK_SIZE_N': 4096}, num_stages=4, num_warps=8),
    ],
    key=['batch_size', 'out_channels', 'height', 'width'],
)
@triton.jit
def fused_conv_sigmoid_mul_kernel(
    weight_ptr, bias_ptr, input_ptr,
    mul_input_ptr,
    output_ptr,
    weight_stride,
    batch_size, out_channels, in_channels, height, width,
    num_programs,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    pid = tl.program_id(0)
    total_elements = batch_size * out_channels * height * width
    
    # Each program processes BLOCK_SIZE_N elements
    start_offset = pid * BLOCK_SIZE_N
    col_offsets = start_offset + tl.arange(0, BLOCK_SIZE_N)
    mask = col_offsets < total_elements
    
    # Compute indices
    stride = out_channels * height * width
    batch_idx = col_offsets // stride
    remainder = col_offsets % stride
    ch_stride = height * width
    out_ch_idx = remainder // ch_stride
    hw_idx = remainder % ch_stride
    
    # Flat index
    flat_idx = batch_idx * stride + out_ch_idx * ch_stride + hw_idx
    
    # Load bias
    bias = tl.load(bias_ptr + out_ch_idx, mask=mask)
    
    # Convolution (sum over input channels)
    conv_result = tl.zeros_like(bias)
    for c in range(in_channels):
        w_ptr = weight_ptr + out_ch_idx * weight_stride + c
        weight_val = tl.load(w_ptr, mask=mask)
        inp_idx = batch_idx * in_channels + c
        input_val = tl.load(input_ptr + inp_idx, mask=mask)
        conv_result = conv_result + weight_val * input_val
    
    # Add bias and apply sigmoid
    conv_result = conv_result + bias
    sigmoid_result = 1.0 / (1.0 + tl.exp(-conv_result))
    
    # Multiply
    mul_input = tl.load(mul_input_ptr + flat_idx, mask=mask)
    result = mul_input * sigmoid_result
    
    # Store
    tl.store(output_ptr + flat_idx, result, mask=mask)


@torch.fx.wrap
def fused_conv_sigmoid_mul_kernel_wrapper(bias, weight, mul_input, conv_input):
    """
    Fused Conv2D + Sigmoid + Multiply kernel.
    
    Args:
        bias: [out_channels]
        weight: [out_channels, in_channels, 1, 1]
        mul_input: [batch, out_channels, height, width]
        conv_input: [batch, in_channels, 1, 1]
    
    Returns:
        Element-wise multiplied result [batch, out_channels, height, width]
    """
    batch_size = mul_input.shape[0]
    out_channels = mul_input.shape[1]
    height = mul_input.shape[2]
    width = mul_input.shape[3]
    in_channels = conv_input.shape[1]
    
    # Output shape
    output_shape = (batch_size, out_channels, height, width)
    output = torch.empty(output_shape, device=mul_input.device, dtype=mul_input.dtype)
    
    # Calculate total elements
    total_elements = batch_size * out_channels * height * width
    
    # Use num_programs proportional to workload but capped at 256
    # This balances parallelism with kernel launch overhead
    num_programs = min(max(1, (total_elements + 511) // 512), 256)
    
    # Launch kernel with autotuning
    grid = (num_programs,)
    fused_conv_sigmoid_mul_kernel[grid](
        weight_ptr=weight,
        bias_ptr=bias,
        input_ptr=conv_input,
        mul_input_ptr=mul_input,
        output_ptr=output,
        weight_stride=weight.stride(0),
        batch_size=batch_size,
        out_channels=out_channels,
        in_channels=in_channels,
        height=height,
        width=width,
        num_programs=num_programs,
    )
    
    return output


def replacement_func():
    return fused_conv_sigmoid_mul_kernel_wrapper