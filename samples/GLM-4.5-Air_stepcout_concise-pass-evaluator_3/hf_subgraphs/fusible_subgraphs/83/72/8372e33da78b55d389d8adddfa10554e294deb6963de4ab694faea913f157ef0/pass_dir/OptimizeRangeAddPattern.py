import torch
import triton
import triton.language as tl

def pattern(in_0, in_1, in_2):
    """Optimize the range generation + scalar multiplication + addition pattern"""
    tmp_2 = torch.arange(start=0, end=in_2, device=torch.device('cuda'))
    tmp_3 = tmp_2 * in_1
    tmp_4 = tmp_3.view((1,))
    tmp_5 = tmp_4.unsqueeze(-1)
    tmp_6 = tmp_5 + in_0
    result = tmp_6.view(-1)
    return result

def replacement_args(in_0, in_1, in_2):
    """Extract arguments for the replacement kernel"""
    return (in_0, in_1, in_2)

@triton.jit
def fused_range_add_kernel(
    input_ptr,
    scalar_ptr,
    range_end_ptr,
    output_ptr,
    input_size_0,
    input_size_1,
    range_val,
    BLOCK_SIZE: tl.constexpr,
):
    """Fused kernel that combines range generation, scalar multiplication, and addition"""
    # Each program processes a contiguous block of data
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < (input_size_0 * input_size_1)
    
    # Load input data (flattened from input shape)
    input_data = tl.load(input_ptr + offsets, mask=mask, other=0)
    
    # Generate range values and apply scalar multiplication directly
    # Note: triton.language.arange doesn't exist, so we calculate directly
    range_vals = offsets + range_val
    
    # Perform fused computation: input_data + range_vals * scalar
    scalar = tl.load(scalar_ptr)
    result = input_data + range_vals * scalar
    
    # Store result
    tl.store(output_ptr + offsets, result, mask=mask)

@torch.fx.wrap
def fused_range_add_torch(in_0, in_1, in_2):
    """PyTorch wrapper for the fused kernel"""
    # Get input tensor shape
    in_shape = in_0.shape
    
    # Ensure in_1 is a scalar (extract value if tensor)
    if torch.is_tensor(in_1):
        scalar_val = in_1.item()
    else:
        scalar_val = in_1
    
    # Ensure in_2 is a scalar (extract value if tensor)
    if torch.is_tensor(in_2):
        range_start = 0
        range_end = in_2.item()
    else:
        range_start = 0
        range_end = in_2
    
    # Calculate flattened size
    total_elements = in_shape[0] * in_shape[1]
    
    # Determine block size
    BLOCK_SIZE = 1024
    num_programs = (total_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Create output tensor
    output = torch.empty(total_elements, dtype=in_0.dtype, device=in_0.device)
    
    # Launch kernel
    fused_range_add_kernel[(num_programs,)](
        input_ptr=in_0.flatten(),
        scalar_ptr=scalar_val,
        range_end_ptr=range_end,
        output_ptr=output,
        input_size_0=in_shape[0],
        input_size_1=in_shape[1],
        range_val=range_start,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return output

def replacement_func():
    """Return the optimized function"""
    return fused_range_add_torch