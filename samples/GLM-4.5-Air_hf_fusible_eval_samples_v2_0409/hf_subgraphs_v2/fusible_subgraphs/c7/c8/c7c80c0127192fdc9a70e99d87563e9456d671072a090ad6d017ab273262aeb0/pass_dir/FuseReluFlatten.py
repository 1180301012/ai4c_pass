import torch
import triton
import triton.language as tl

# Pattern matching function - matches ReLU + Flatten sequence
def pattern(in_0):
    """
    Pattern matches ReLU (inplace) followed by Flatten(1, -1)
    """
    tmp_0 = torch.nn.functional.relu(in_0, inplace = True)
    tmp_1 = tmp_0.flatten(1, -1)
    return tmp_0, tmp_1

# Argument extraction function
def replacement_args(in_0):
    """
    Extract input tensor for the fused operation
    """
    return (in_0,)

# Optimized kernel implementation
@triton.jit
def fused_relu_flatten_kernel(
    input_ptr,
    output_ptr,
    n_elements: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fused ReLU + Flatten kernel
    Performs ReLU activation and handles flattening layout in memory
    """
    # Each program handles a contiguous block of data
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load input data
    x = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    
    # Apply ReLU activation
    out = tl.maximum(x, 0.0)
    
    # Store result
    tl.store(output_ptr + offsets, out, mask=mask)

@torch.fx.wrap
def fused_relu_flatten(input_tensor):
    """
    Wrapper function for launching the fused ReLU + Flatten kernel
    """
    # Calculate total number of elements
    n_elements = input_tensor.numel()
    
    # Set up block size and grid
    BLOCK_SIZE = 1024
    num_programs = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Create output tensor with same dtype as input, but flattened shape
    # For tensors like [N, C, H, W], flattening from dim 1 gives [N, C*H*W]
    original_shape = input_tensor.shape
    flattened_shape = (original_shape[0], -1)
    output = torch.empty(flattened_shape, dtype=input_tensor.dtype, device=input_tensor.device)
    
    # Launch kernel
    fused_relu_flatten_kernel[(num_programs,)](
        input_ptr=input_tensor,
        output_ptr=output,
        n_elements=n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return output

# Replacement function - returns the fused kernel wrapper
def replacement_func():
    """
    Returns reference to the fused ReLU + Flatten implementation
    """
    return fused_relu_flatten