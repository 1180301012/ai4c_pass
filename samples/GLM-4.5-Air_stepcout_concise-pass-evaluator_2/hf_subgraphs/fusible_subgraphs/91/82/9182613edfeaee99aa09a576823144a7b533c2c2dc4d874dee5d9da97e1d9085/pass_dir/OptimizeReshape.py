import torch
import triton
import triton.language as tl
import math

def pattern(x, shape):
    result = x.reshape(shape)
    return result

def replacement_args(x, shape):
    return (x, shape)

@triton.jit
def optimized_reshape_kernel(
    input_ptr,
    output_ptr,
    input_total_elements,
    output_total_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offset = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offset < output_total_elements
    
    # Simple linear copy for reshape operations
    # This works well when the linear memory can be preserved
    input_idx = offset
    if offset < input_total_elements:
        input_data = tl.load(input_ptr + offset, mask=mask, other=0.0)
        tl.store(output_ptr + offset, input_data, mask=mask)

@torch.fx.wrap
def optimized_reshape(x, shape):
    """
    Optimized reshape that checks if contiguous copy is actually needed.
    For simple reshape cases where memory layout allows direct reshaping,
    we can avoid the expensive contiguous operation.
    """
    # Check if source tensor is already contiguous and has optimal layout
    if not x.is_contiguous():
        # If source is not contiguous, make it contiguous first
        x = x.contiguous()
    
    # Create output tensor
    output = torch.empty(shape, dtype=x.dtype, device=x.device)
    
    total_input_elements = x.numel()
    total_output_elements = output.numel()
    
    # If input and output have the same total elements and source is contiguous,
    # we can do a direct memory copy without full reshape overhead
    if total_input_elements == total_output_elements and x.is_contiguous():
        # For simple reshape cases, use optimized kernel
        if total_output_elements > 10000:  # Only use kernel for larger tensors
            BLOCK_SIZE = 1024
            num_programs = math.ceil(total_output_elements / BLOCK_SIZE)
            
            optimized_reshape_kernel[(num_programs,)](
                input_ptr=x,
                output_ptr=output,
                input_total_elements=total_input_elements,
                output_total_elements=total_output_elements,
                BLOCK_SIZE=BLOCK_SIZE,
            )
        else:
            # For small tensors, use standard reshape which is often faster
            output = x.reshape(shape)
    else:
        # For complex reshape cases, use standard PyTorch reshape
        output = x.reshape(shape)
    
    return output

def replacement_func():
    return optimized_reshape