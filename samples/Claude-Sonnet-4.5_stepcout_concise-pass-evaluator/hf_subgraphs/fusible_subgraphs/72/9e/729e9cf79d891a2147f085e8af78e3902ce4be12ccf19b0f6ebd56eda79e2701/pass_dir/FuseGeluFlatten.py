import torch
import triton
import triton.language as tl
import math

# Pattern matching function
def pattern(in_0):
    """ 
    Match GELU followed by flatten operation
    """
    tmp_0 = torch.nn.functional.gelu(in_0, approximate='none')
    tmp_1 = tmp_0.flatten(1, -1)
    return (tmp_1,)

# Argument extraction function
def replacement_args(in_0):
    return (in_0,)

# Optimized Triton kernel for fused GELU + flatten
@triton.jit
def gelu_flatten_kernel(
    input_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a contiguous block of data
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load input
    x = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    
    # Compute GELU: 0.5 * x * (1 + erf(x / sqrt(2)))
    sqrt_2 = 1.4142135623730951
    erf_arg = x / sqrt_2
    erf_val = tl.math.erf(erf_arg)
    gelu_out = 0.5 * x * (1.0 + erf_val)
    
    # Store output (flatten is just a reshape, data layout is the same)
    tl.store(output_ptr + offsets, gelu_out, mask=mask)

# Kernel wrapper
@torch.fx.wrap
def fused_gelu_flatten(input_tensor):
    # Get input shape
    input_shape = input_tensor.shape
    
    # Calculate output shape for flatten(1, -1)
    batch_size = input_shape[0]
    flat_size = 1
    for dim in input_shape[1:]:
        flat_size *= dim
    output_shape = (batch_size, flat_size)
    
    # Allocate output tensor
    output = torch.empty(output_shape, dtype=input_tensor.dtype, device=input_tensor.device)
    
    # Total number of elements
    n_elements = input_tensor.numel()
    
    # Choose block size - smaller for small inputs
    BLOCK_SIZE = 256
    num_blocks = triton.cdiv(n_elements, BLOCK_SIZE)
    
    # Launch kernel
    gelu_flatten_kernel[(num_blocks,)](
        input_ptr=input_tensor,
        output_ptr=output,
        n_elements=n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return output

# Replacement function (returns function reference)
def replacement_func():
    return fused_gelu_flatten