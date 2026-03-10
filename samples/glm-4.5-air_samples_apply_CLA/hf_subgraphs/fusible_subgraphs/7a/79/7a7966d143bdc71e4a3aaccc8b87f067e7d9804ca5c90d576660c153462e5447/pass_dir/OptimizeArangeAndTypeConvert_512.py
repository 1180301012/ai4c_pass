import torch
import triton
import triton.language as tl
from torch import device

def pattern(in_0):
    tmp_0 = in_0
    # Match flexible arange operation
    tmp_1 = torch.arange(0, 512)  # Basic arange, device will be handled separately
    tmp_2 = tmp_0.to(dtype=torch.bool)
    return tmp_1, tmp_2

def replacement_args(in_0):
    return (in_0, 512)  # Hardcoded arange size for this pass

@triton.jit
def arange_and_convert_kernel(
    output_arange_ptr,
    output_bool_ptr,
    input_ptr,
    arange_size,
    input_size,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    
    # Process arange generation
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < arange_size
    
    # Generate arange values directly on GPU
    arange_values = offsets.to(tl.int64)
    tl.store(output_arange_ptr + offsets, arange_values, mask=mask)
    
    # Process boolean conversion
    input_offsets = block_start + tl.arange(0, BLOCK_SIZE)
    input_mask = input_offsets < input_size
    
    # Load input values and convert to bool (non-zero -> True)
    input_vals = tl.load(input_ptr + input_offsets, mask=input_mask, other=0)
    bool_vals = input_vals != 0  # Convert to bool
    tl.store(output_bool_ptr + input_offsets, bool_vals, mask=input_mask)

@torch.fx.wrap
def optimized_arange_and_convert(input_tensor, arange_size):
    # Get input tensor properties
    input_shape = input_tensor.shape
    input_size = input_tensor.numel()
    input_device = input_tensor.device
    input_dtype = input_tensor.dtype
    
    # Create output tensors
    arange_tensor = torch.empty(arange_size, dtype=torch.int64, device=input_device)
    bool_tensor = torch.empty(input_shape, dtype=torch.bool, device=input_device)
    
    # Launch Triton kernel
    BLOCK_SIZE = 1024
    num_programs_arange = (arange_size + BLOCK_SIZE - 1) // BLOCK_SIZE
    num_programs_input = (input_size + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Use launches based on the larger dimension
    grid = (max(num_programs_arange, num_programs_input),)
    
    arange_and_convert_kernel[grid](
        output_arange_ptr=arange_tensor,
        output_bool_ptr=bool_tensor,
        input_ptr=input_tensor,
        arange_size=arange_size,
        input_size=input_size,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return arange_tensor, bool_tensor

def replacement_func():
    return optimized_arange_and_convert