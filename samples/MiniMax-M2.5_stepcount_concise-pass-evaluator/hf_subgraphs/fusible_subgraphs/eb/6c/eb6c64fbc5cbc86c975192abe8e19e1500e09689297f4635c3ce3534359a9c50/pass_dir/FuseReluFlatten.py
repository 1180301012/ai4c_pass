import torch
import triton
import triton.language as tl


# Pattern matching function
def pattern(in_0):
    """
    Match the pattern: relu(in_0, inplace=True) followed by flatten(1, -1)
    The pattern mirrors the model.py computation exactly.
    """
    tmp_0 = torch.nn.functional.relu(in_0, inplace=True)
    tmp_1 = tmp_0.flatten(1, -1)
    return tmp_1


# Argument extraction function
def replacement_args(in_0):
    return (in_0,)


# Optimized Triton kernel that fuses ReLU and flatten
@triton.jit
def relu_flatten_kernel(
    input_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a contiguous block of data of size BLOCK_SIZE
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load input values
    x = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    
    # Apply ReLU: max(0, x)
    # Using tl.where for in-place semantics equivalent
    x = tl.where(x > 0, x, 0.0)
    
    # Store the result
    tl.store(output_ptr + offsets, x, mask=mask)


@torch.fx.wrap
def relu_flatten_wrapper(x):
    """
    Wrapper function that launches the Triton kernel.
    This fuses ReLU and flatten(1, -1) into a single operation.
    """
    # Get the total number of elements (batch * channels for the 2D output)
    # Input is [B, C, 1, 1], output is [B, C]
    # The flatten(1, -1) essentially just reshapes to [B, C]
    n_elements = x.numel()
    
    # For small tensors, use a simple kernel launch
    BLOCK_SIZE = 1024
    num_programs = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Allocate output tensor with shape [B, C]
    # Get the first two dimensions as the output shape
    output_shape = (x.shape[0], x.shape[1])
    output = torch.empty(output_shape, dtype=x.dtype, device=x.device)
    
    # Launch the kernel
    relu_flatten_kernel[(num_programs,)](
        input_ptr=x,
        output_ptr=output,
        n_elements=n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return output


def replacement_func():
    return relu_flatten_wrapper