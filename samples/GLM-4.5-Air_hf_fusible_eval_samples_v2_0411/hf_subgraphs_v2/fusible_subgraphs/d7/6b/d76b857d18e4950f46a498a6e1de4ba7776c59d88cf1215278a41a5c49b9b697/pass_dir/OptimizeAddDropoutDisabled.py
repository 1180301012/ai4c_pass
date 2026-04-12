import torch
import triton
import triton.language as tl

def pattern(in_4, in_3):
    """Match addition followed by disabled dropout2d (training=False, p=0.1)"""
    tmp_3 = in_4 + in_3
    tmp_4 = torch.nn.functional.dropout2d(tmp_3, 0.1, False, False)
    return tmp_4

def replacement_args(in_4, in_3):
    return (in_4, in_3)

@triton.jit
def add_kernel(
    x_ptr,
    y_ptr,
    out_ptr,
    batch_size,
    channels,
    height,
    width,
    BLOCK_SIZE: tl.constexpr,
):
    """Element-wise addition kernel"""
    # Calculate thread position
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # Create mask for valid elements
    total_elements = batch_size * channels * height * width
    mask = offsets < total_elements
    
    # Load both tensors and perform addition
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = tl.load(y_ptr + offsets, mask=mask, other=0.0)
    out = x + y
    
    # Store result
    tl.store(out_ptr + offsets, out, mask=mask)

@torch.fx.wrap
def optimized_addition_dropout_disabled(x, y):
    """Optimized function for addition when dropout is disabled"""
    batch_size, channels, height, width = x.shape
    total_elements = batch_size * channels * height * width
    
    # Output tensor
    result = torch.empty_like(x)
    
    # Configure block size for optimal GPU utilization
    BLOCK_SIZE = 1024
    
    # Calculate number of programs needed
    num_programs = (total_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Launch addition kernel
    add_kernel[(num_programs,)](
        x_ptr=x,
        y_ptr=y,
        out_ptr=result,
        batch_size=batch_size,
        channels=channels,
        height=height,
        width=width,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return result

def replacement_func():
    return optimized_addition_dropout_disabled