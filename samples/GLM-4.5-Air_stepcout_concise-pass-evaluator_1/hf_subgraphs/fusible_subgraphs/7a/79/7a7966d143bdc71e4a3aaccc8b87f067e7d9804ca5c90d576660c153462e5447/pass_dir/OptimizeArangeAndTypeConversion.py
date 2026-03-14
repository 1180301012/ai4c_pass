import torch
from torch import device
import triton
import triton.language as tl

# Pattern matching function
def pattern(input_tensor):
    """
    Match the pattern: create arange + convert input to boolean
    """
    tmp_0 = input_tensor
    tmp_1 = torch.arange(0, 64, device=device(type='cuda', index=0))
    tmp_2 = tmp_0.to(device=device(type='cuda', index=0), dtype=torch.bool)
    return (tmp_1, tmp_2)

# Argument extraction function
def replacement_args(input_tensor):
    """
    Extract arguments needed for the replacement
    """
    return (input_tensor,)

# Optimized kernel for size-specific arange generation and type conversion
@triton.jit
def arange_and_convert_kernel_64(
    input_ptr,
    arange_out_ptr,
    convert_out_ptr,
    input_shape_0,
    input_shape_1,
    BLOCK_SIZE: tl.constexpr,
):
    """Kernel that generates arange=64 and performs type conversion in parallel"""
    # Handle type conversion (work distribution for each output element)
    total_elements = input_shape_0 * input_shape_1
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < total_elements
    
    # Load input data and convert to boolean
    input_vals = tl.load(input_ptr + offsets, mask=mask, other=0)
    # Convert to boolean by checking non-zero (int64 -> bool)
    bool_vals = input_vals != 0
    tl.store(convert_out_ptr + offsets, bool_vals, mask=mask)

@torch.fx.wrap
def optimized_arange_and_convert_64(input_tensor):
    """
    Optimized function that generates arange=64 and converts input to boolean
    """
    input_shape = input_tensor.shape
    input_shape_0, input_shape_1 = input_shape[0], input_shape[1] if len(input_shape) > 1 else 1
    
    # Create output tensors
    arange_out = torch.arange(64, dtype=torch.int64, device='cuda')
    convert_out = torch.empty(input_shape_0 * input_shape_1, dtype=torch.bool, device='cuda')
    
    # Reshape input to 1D for easier processing
    input_flat = input_tensor.reshape(-1)
    
    BLOCK_SIZE = 1024
    num_programs = (input_shape_0 * input_shape_1 + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    arange_and_convert_kernel_64[(num_programs,)](
        input_ptr=input_flat,
        arange_out_ptr=arange_out,
        convert_out_ptr=convert_out,
        input_shape_0=input_shape_0,
        input_shape_1=input_shape_1,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    # Reshape back to original input shape
    convert_out = convert_out.reshape(input_shape)
    
    return arange_out, convert_out

# Replacement function (returns function reference)
def replacement_func():
    """
    Return the optimized function
    """
    return optimized_arange_and_convert_64