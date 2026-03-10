import torch
import triton
import triton.language as tl

def pattern(conv_output, in_2):
    tmp_3 = conv_output + 1.0
    tmp_4 = tmp_3 / 2.0
    tmp_5 = tmp_4.clamp_(0.0, 1.0)
    tmp_6 = in_2 * tmp_5
    return tmp_6

def replacement_args(conv_output, in_2):
    return (conv_output, in_2)

@triton.jit
def fused_elementwise_kernel(
    conv_output_ptr,
    in_2_ptr,
    output_ptr,
    batch_size, channels, height, width,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles BLOCK_SIZE elements
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # Calculate total size and create mask
    total_elements = batch_size * channels * height * width
    mask = offsets < total_elements
    
    # Load conv output and in_2 values
    conv_vals = tl.load(conv_output_ptr + offsets, mask=mask, other=0.0)
    in_2_vals = tl.load(in_2_ptr + offsets, mask=mask, other=0.0)
    
    # Fuse elementwise operations: add 1.0, div 2.0, clamp [0,1], multiply by in_2
    fused_vals = (conv_vals + 1.0) / 2.0
    fused_vals = tl.maximum(0.0, tl.minimum(1.0, fused_vals))
    fused_vals = fused_vals * in_2_vals
    
    # Store results
    tl.store(output_ptr + offsets, fused_vals, mask=mask)

@torch.fx.wrap
def fused_elementwise_ops(conv_output, in_2):
    # Get input shapes
    batch_size, channels, height, width = conv_output.shape
    
    # Create output tensor
    output = torch.empty(batch_size, channels, height, width, device=conv_output.device, dtype=conv_output.dtype)
    
    # Optimize block size based on total elements
    total_elements = batch_size * channels * height * width
    BLOCK_SIZE = 1024  # Can be autotuned
    
    # Adjust grid size based on total elements
    num_programs = (total_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Launch kernel
    fused_elementwise_kernel[(num_programs,)](
        conv_output_ptr=conv_output,
        in_2_ptr=in_2,
        output_ptr=output,
        batch_size=batch_size, channels=channels, height=height, width=width,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return output

def replacement_func():
    return fused_elementwise_ops