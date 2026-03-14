import torch
import triton
import triton.language as tl

def pattern(input_tensor):
    """
    Match the pattern: expand(3, -1, -1).to(device)
    This can be optimized by directly creating the expanded tensor on the target device
    """
    expanded = input_tensor.expand(3, -1, -1)
    result = expanded.to(device=input_tensor.device)
    return result

def replacement_args(input_tensor):
    return (input_tensor,)

@triton.jit
def create_expanded_kernel(
    input_ptr,
    output_ptr,
    n_elements,
    batch_size: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Create expanded tensor directly on target device
    This avoids the intermediate CPU tensor creation during expand operation
    """
    # Each program handles a contiguous block of data
    pid = tl.program_id(0)
    
    # Calculate the total size for one batch (all expanded copies)
    batch_elements = n_elements * batch_size
    
    # Handle this batch copy
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < batch_elements
    
    # Load input data (will be broadcasted)
    input_data = tl.load(input_ptr, mask=offsets % n_elements < n_elements, other=0)
    
    # Store the expanded copy
    tl.store(output_ptr + offsets, input_data, mask=mask)

@torch.fx.wrap
def create_expanded_on_device(input_tensor, batch_size=3):
    """
    Create expanded tensor directly on target device without intermediate CPU tensor
    """
    # Get input tensor info
    n_elements = input_tensor.numel()
    
    # Calculate output size
    output_size = (batch_size,) + input_tensor.shape
    output_elements = batch_size * n_elements
    
    # Create output tensor on target device
    output = torch.empty(output_size, dtype=input_tensor.dtype, device=input_tensor.device)
    
    if output_elements > 0:
        BLOCK_SIZE = 1024
        num_programs = (output_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
        
        create_expanded_kernel[(num_programs,)](
            input_ptr=input_tensor,
            output_ptr=output,
            n_elements=n_elements,
            batch_size=batch_size,
            BLOCK_SIZE=BLOCK_SIZE,
        )
    
    return output

def replacement_func():
    return create_expanded_on_device