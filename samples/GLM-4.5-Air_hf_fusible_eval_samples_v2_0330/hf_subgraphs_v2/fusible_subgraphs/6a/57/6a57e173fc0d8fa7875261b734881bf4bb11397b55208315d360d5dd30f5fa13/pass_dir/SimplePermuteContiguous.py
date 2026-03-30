import torch
import triton
import triton.language as tl

def pattern(x):
    # Match permute + contiguous sequence
    tmp_3 = x.permute(0, 2, 1, 3)
    tmp_4 = tmp_3.contiguous()
    # Return the observable result
    return tmp_4

def replacement_args(x):
    return (x,)

@triton.jit
def permute_contiguous_kernel(
    input_ptr,           # [batch, channels, height, width]
    output_ptr,          # [batch, height, channels, width]
    batch_size,
    channels,
    height,
    width,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles block_size elements
    linear_idx = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = linear_idx < (batch_size * channels * height * width)
    
    # Calculate coordinates in original tensor
    batch = linear_idx // (channels * height * width)
    residual = linear_idx % (channels * height * width)
    c = residual // (height * width)
    residual = residual % (height * width)
    h = residual // width
    w = residual % width
    
    # Calculate coordinates in permuted tensor [batch, height, channels, width]
    batch_out = batch
    h_out = h
    c_out = c
    w_out = w
    
    # Calculate input offset: [batch, channels, height, width]
    input_offset = batch * channels * height * width + c * height * width + h * width + w
    
    # Calculate output offset: [batch, height, channels, width]
    output_offset = batch_out * height * channels * width + h_out * channels * width + c_out * width + w_out
    
    # Load from input and store to output with permuted dimensions
    input_val = tl.load(input_ptr + input_offset, mask=mask)
    tl.store(output_ptr + output_offset, input_val, mask=mask)

@torch.fx.wrap
def optimized_permute_contiguous(x):
    batch_size, channels, height, width = x.shape
    
    # Create output tensor with permuted layout: [batch, height, channels, width]
    output = torch.empty((batch_size, height, channels, width), dtype=x.dtype, device=x.device)
    
    # Calculate total elements and set up grid
    total_elements = batch_size * channels * height * width
    grid = (triton.cdiv(total_elements, 1024),)
    
    # Launch kernel
    permute_contiguous_kernel[grid](
        input_ptr=x,
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