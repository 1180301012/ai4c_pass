import torch
import triton
import triton.language as tl

def pattern(in_1, tmp_1):
    tmp_2 = in_1 * tmp_1
    tmp_3 = in_1 + tmp_2
    return tmp_3

def replacement_args(in_1, tmp_1):
    return (in_1, tmp_1)

@triton.jit
def fused_mul_add_kernel(
    input_ptr,
    scale_ptr,
    output_ptr,
    batch_size,
    channels,
    height,
    width,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    
    # Each program handles a block of elements
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < (batch_size * channels * height * width)
    
    # Load input tensor values
    input_vals = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    
    # Load scale values (1.0 + tmp_1) - need to handle broadcasting
    # For spatial broadcasting, each element gets the same scale value for its channel
    flat_offsets = offsets
    batch_idx = flat_offsets // (channels * height * width)
    remainder = flat_offsets % (channels * height * width)
    channel_idx = remainder // (height * width)
    spatial_idx = remainder % (height * width)
    
    # Load scale values from the scale tensor (should be broadcasted)
    scale_offset = channel_idx  # tmp_1 is [1, 512, 1, 1] so we use channel indexing
    scale_vals = tl.load(scale_ptr + scale_offset, mask=(scale_offset < channels), other=1.0)
    
    # Apply fused computation: input * (1.0 + scale)
    # Scale values should be added to 1.0 first
    scale_plus_one = scale_vals + 1.0
    output_vals = input_vals * scale_plus_one
    
    # Store results
    tl.store(output_ptr + offsets, output_vals, mask=mask)

@torch.fx.wrap
def fused_elemwise_mul_add(in_1, tmp_1):
    # Get tensor shapes
    input_shape = in_1.shape    # [1, 512, 64, 64]
    scale_shape = tmp_1.shape    # [1, 512, 1, 1]
    
    batch_size, channels, height, width = input_shape
    
    total_elements = batch_size * channels * height * width
    BLOCK_SIZE = 1024  # Optimize for better GPU occupancy
    
    num_programs = (total_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Output tensor
    output = torch.empty_like(in_1)
    
    # Launch kernel
    fused_mul_add_kernel[(num_programs,)](
        input_ptr=in_1,
        scale_ptr=tmp_1,
        output_ptr=output,
        batch_size=batch_size,
        channels=channels,
        height=height,
        width=width,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return output

def replacement_func():
    return fused_elemwise_mul_add