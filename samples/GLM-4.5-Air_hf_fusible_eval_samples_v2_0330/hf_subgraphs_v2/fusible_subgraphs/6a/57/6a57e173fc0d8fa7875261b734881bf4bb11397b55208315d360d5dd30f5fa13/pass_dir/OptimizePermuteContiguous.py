import torch
import triton
import triton.language as tl

def pattern(in_3):
    # Match the permute operation
    tmp_3 = in_3.permute(0, 2, 1, 3)
    # Match the contiguous operation
    tmp_4 = tmp_3.contiguous()
    # Return the observable result
    return tmp_4

def replacement_args(in_3):
    return (in_3,)

@triton.jit
def optimized_permute_contiguous_kernel(
    input_ptr,           # [batch, channels, height, width] (original layout)
    output_ptr,          # [batch, height, channels, width] (permuted layout)
    batch_size,
    channels,
    height,
    width,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles one element in the output tensor
    linear_idx = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    batch = linear_idx // (height * channels * width)
    residual = linear_idx % (height * channels * width)
    h = residual // (channels * width)
    residual = residual % (channels * width)
    c = residual // width
    w = residual % width
    
    # Create mask for valid elements
    mask = (batch < batch_size) & (h < height) & (c < channels) & (w < width)
    
    # Calculate input offsets (original layout: [batch, channels, height, width])
    input_offset = batch * channels * height * width + c * height * width + h * width + w
    
    # Calculate output offsets (permuted layout: [batch, height, channels, width])
    output_offset = batch * height * channels * width + h * channels * width + c * width + w
    
    # Load from input and store to output with permuted dimensions
    # This effectively: output[batch, h, c, w] = input[batch, c, h, w]
    input_val = tl.load(input_ptr + input_offset, mask=mask)
    tl.store(output_ptr + output_offset, input_val, mask=mask)

@torch.fx.wrap
def optimized_permute_contiguous(input_tensor):
    # Get input dimensions
    batch_size, channels, height, width = input_tensor.shape
    
    # Create output tensor with permuted layout: [batch, height, channels, width]
    output = torch.empty((batch_size, height, channels, width), dtype=input_tensor.dtype, device=input_tensor.device)
    
    # Calculate total number of elements
    total_elements = batch_size * height * channels * width
    
    # Set up grid dimensions
    grid = (triton.cdiv(total_elements, 1024),)
    
    # Launch kernel with optimized memory access
    optimized_permute_contiguous_kernel[grid](
        input_ptr=input_tensor,
        output_ptr=output,
        batch_size=batch_size,
        channels=channels,
        height=height,
        width=width,
        BLOCK_SIZE=1024,
    )
    
    return output

def replacement_func():
    return optimized_permute_contiguous