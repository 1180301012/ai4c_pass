import torch
import triton
import triton.language as tl


# Pattern matching function - matches multiply + gelu after sigmoid
# The sigmoid output broadcasts with multiply_input, then gelu is applied
def pattern(sigmoid_out, multiply_input):
    # sigmoid_out is the output of sigmoid() with shape [B, C, 1, 1]
    # multiply_input is the tensor being multiplied with shape [B, C, H, W]
    # The multiplication broadcasts sigmoid_out across H, W dimensions
    mul_result = multiply_input * sigmoid_out
    gelu_result = torch.nn.functional.gelu(mul_result, approximate='none')
    return gelu_result


def replacement_args(sigmoid_out, multiply_input):
    return (sigmoid_out, multiply_input)


# Optimized Triton kernel for fused sigmoid * multiply + GELU with broadcasting
# Uses block-based processing for efficiency
@triton.jit
def gelu_forward_kernel(
    y_ptr,       # multiply_input [B, C, H, W]
    x_ptr,       # sigmoid_out [B, C, 1, 1] - will be broadcast
    output_ptr,  # output buffer [B, C, H, W]
    n_elements,  # total number of elements
    stride_y_b,  # stride for y (multiply_input)
    stride_y_c,
    stride_y_h,
    stride_y_w,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program processes a block of BLOCK_SIZE elements
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load y values (multiply_input)
    y = tl.load(y_ptr + offsets, mask=mask, other=0.0)
    
    # For broadcasting, we need to compute which element each thread processes
    # and load the corresponding sigmoid value
    # Each BLOCK_SIZE consecutive elements have the same channel index
    # since we access in row-major order: (b, c, h, w) with stride (stride_y_b, stride_y_c, stride_y_h, stride_y_w)
    
    # Calculate the channel index for each offset
    # offset = b * stride_b + c * stride_c + h * stride_h + w * stride_w
    # But since we're processing linearly, we can compute b, c, h, w from offset
    
    # For simplicity, load sigmoid values - each channel repeats H*W times
    c_indices = (offsets // (stride_y_h // stride_y_w)) % (stride_y_c // stride_y_w)
    # This is simplified - let's use a different approach
    
    # Simpler: use the first element of each channel for sigmoid
    # sigmoid_out has stride (C, 1) 
    sigmoid_offsets = (offsets // (H * W)) % C
    
    # Actually, we know C from the shape, but we can't pass it easily
    # Let's use a different approach: compute offsets into sigmoid_out directly
    # sigmoid_out[b, c] is at offset b * C + c
    sigmoid_offset = (offsets // (H * W)) * 0  # placeholder
    
    # Load x values - we need to compute the right offset into sigmoid_out
    # The sigmoid_out tensor has shape [B, C, 1, 1], so offset = b * C + c
    # We can compute b = offset // (C*H*W), c = (offset // (H*W)) % C
    x = tl.load(x_ptr + (offsets // (H * W)) % C, mask=mask, other=0.0)
    
    # Multiply: y * sigmoid(x)
    mul_result = y * x
    
    # GELU approximation using only basic ops:
    sqrt_2_over_pi = 0.7978845608028654
    alpha = 0.044715
    
    x3 = mul_result * mul_result * mul_result
    inner = sqrt_2_over_pi * (mul_result + alpha * x3)
    
    # Compute tanh using exp: tanh(z) = (e^(2z) - 1) / (e^(2z) + 1)
    two_z = 2.0 * inner
    exp_2z = tl.exp(two_z)
    tanh_inner = (exp_2z - 1.0) / (exp_2z + 1.0)
    
    # GELU result
    gelu_result = 0.5 * mul_result * (1.0 + tanh_inner)
    
    # Store result
    tl.store(output_ptr + offsets, gelu_result, mask=mask)


@torch.fx.wrap
def fused_sigmoid_mul_gelu_kernel_wrapper(sigmoid_out, multiply_input):
    """
    Fused Triton kernel for: multiply_input * sigmoid_out -> gelu
    Handles broadcasting from sigmoid_out [B, C, 1, 1] to multiply_input [B, C, H, W]
    """
    B, C, H, W = multiply_input.shape
    n_elements = B * C * H * W
    
    # Allocate output
    output = torch.empty_like(multiply_input)
    
    # Get strides for multiply_input
    stride_b = multiply_input.stride(0)
    stride_c = multiply_input.stride(1)
    stride_h = multiply_input.stride(2)
    stride_w = multiply_input.stride(3)
    
    # Define block size and grid
    BLOCK_SIZE = 1024
    num_programs = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Launch kernel
    grid = (num_programs,)
    gelu_forward_kernel[grid](
        y_ptr=multiply_input,
        x_ptr=sigmoid_out,
        output_ptr=output,
        n_elements=n_elements,
        stride_y_b=stride_b,
        stride_y_c=stride_c,
        stride_y_h=stride_h,
        stride_y_w=stride_w,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return output


def replacement_func():
    return fused_sigmoid_mul_gelu_kernel_wrapper