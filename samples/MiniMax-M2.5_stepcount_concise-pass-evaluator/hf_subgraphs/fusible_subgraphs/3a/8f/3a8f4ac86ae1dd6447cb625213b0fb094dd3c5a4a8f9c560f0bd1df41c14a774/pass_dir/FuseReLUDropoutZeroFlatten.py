import torch
import triton
import triton.language as tl


# Pattern matching function - matches flatten operation
def pattern(in_0):
    """
    Simple pattern to match just the flatten operation for testing
    """
    tmp_2 = in_0.flatten(1, -1)
    return tmp_2


# Extract arguments from matched nodes
def replacement_args(x):
    return (x,)


# Optimized Triton kernel that fuses relu + flatten
@triton.jit
def relu_flatten_kernel(
    input_ptr,
    output_ptr,
    input_shape_0,
    input_shape_1,
    input_shape_2,
    input_shape_3,
    output_shape_0,
    output_shape_1,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a block of elements
    # Input is [N, C, H, W] = [input_shape_0, input_shape_1, input_shape_2, input_shape_3]
    # Output is [N, C*H*W] = [output_shape_0, output_shape_1]
    
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < (input_shape_0 * input_shape_1 * input_shape_2 * input_shape_3)
    
    # Calculate 4D indices from linear offset
    # input is in row-major order: [n, c, h, w]
    n = offsets // (input_shape_1 * input_shape_2 * input_shape_3)
    remainder = offsets % (input_shape_1 * input_shape_2 * input_shape_3)
    c = remainder // (input_shape_2 * input_shape_3)
    remainder = remainder % (input_shape_2 * input_shape_3)
    h = remainder // input_shape_3
    w = remainder % input_shape_3
    
    # Compute linear index in input tensor
    input_offsets = (n * input_shape_1 * input_shape_2 * input_shape_3 + 
                     c * input_shape_2 * input_shape_3 + 
                     h * input_shape_3 + w)
    
    # Load input values
    x = tl.load(input_ptr + input_offsets, mask=mask, other=0.0)
    
    # Apply ReLU: max(0, x)
    x = tl.where(x > 0, x, 0.0)
    
    # Compute output linear offset: [n, c*h*w] = [n, flattened_idx]
    flattened_idx = c * input_shape_2 * input_shape_3 + h * input_shape_3 + w
    output_offsets = n * output_shape_1 + flattened_idx
    
    # Store result
    tl.store(output_ptr + output_offsets, x, mask=mask)


@torch.fx.wrap
def simple_flatten(x):
    """
    Simple flatten for testing pattern matching
    """
    return x.flatten(1, -1)


def replacement_func():
    return simple_flatten