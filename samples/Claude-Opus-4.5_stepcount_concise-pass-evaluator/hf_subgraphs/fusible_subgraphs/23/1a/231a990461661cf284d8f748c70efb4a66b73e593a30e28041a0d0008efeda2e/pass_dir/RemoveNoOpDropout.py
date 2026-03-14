import torch
import triton
import triton.language as tl

# Pattern to match no-op dropout (probability=0.0)
# This pattern is identical across ALL three graphs
def pattern(input_tensor):
    result = torch.nn.functional.dropout(input_tensor, 0.0, False, False)
    return result

def replacement_args(input_tensor):
    return (input_tensor,)

@triton.jit
def copy_kernel(
    input_ptr, output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(input_ptr + offsets, mask=mask)
    tl.store(output_ptr + offsets, x, mask=mask)

@torch.fx.wrap
def fast_identity(input_tensor):
    # For no-op dropout, just return the input tensor directly
    # This avoids any copy overhead
    return input_tensor

def replacement_func():
    return fast_identity