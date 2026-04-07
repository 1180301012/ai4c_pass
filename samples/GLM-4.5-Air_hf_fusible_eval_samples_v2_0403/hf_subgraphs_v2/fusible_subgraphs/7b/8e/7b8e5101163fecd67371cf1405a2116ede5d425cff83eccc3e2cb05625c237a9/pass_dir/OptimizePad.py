import torch
import triton
import triton.language as tl

def pattern(tmp_4):
    """Pattern: Pad operation with (0,1,0,1) padding"""
    tmp_5 = torch.nn.functional.pad(tmp_4, (0, 1, 0, 1), 'constant', None)
    return tmp_5

def replacement_args(tmp_4):
    return (tmp_4,)

@triton.jit
def optimized_pad_kernel(
    input_ptr,
    output_ptr,
    n_elements,
    height,
    width,
    BLOCK_SIZE: tl.constexpr,
):
    """Optimized pad kernel for (0,1,0,1) padding pattern"""
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Calculate output coordinates
    out_coords = offsets
    batch_idx = out_coords // (height * (width + 1))
    remaining = out_coords % (height * (width + 1))
    h_idx = remaining // (width + 1)
    w_idx = remaining % (width + 1)
    
    # Convert to input coordinates
    input_coords = out_coords + (tl.where(w_idx == width, 1, 0) + tl.where(h_idx == height, (width + 1), 0))
    
    # Load input with bounds checking
    input_vals = tl.load(input_ptr + input_coords, mask=(input_coords < n_elements), other=0.0)
    
    # Store output
    tl.store(output_ptr + out_coords, input_vals, mask=mask)

@torch.fx.wrap
def optimized_pad(x):
    """Optimized pad operation for (0,1,0,1) pattern"""
    batch, channels, height, width = x.shape
    output_shape = (batch, channels, height + 1, width + 1)
    
    # Calculate total elements
    N_input = x.numel()
    N_output = batch * channels * (height + 1) * (width + 1)
    
    BLOCK_SIZE = 1024
    num_programs = (N_output + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    out = torch.empty(output_shape, dtype=x.dtype, device=x.device)
    
    optimized_pad_kernel[(num_programs,)](
        input_ptr=x,
        output_ptr=out,
        n_elements=N_input,
        height=height,
        width=width,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out

def replacement_func():
    return optimized_pad