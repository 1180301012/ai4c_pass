import torch
import triton
import triton.language as tl

def pattern(input_tensor):
    """Match SiLU activation pattern"""
    return torch.nn.functional.silu(input_tensor, inplace=True)

def replacement_args(input_tensor):
    return (input_tensor,)

@triton.jit
def silu_kernel(
    input_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    x = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    # SiLU: x * sigmoid(x)
    sigmoid_x = 1.0 / (1.0 + tl.exp(-x))
    out = x * sigmoid_x
    tl.store(output_ptr + offsets, out, mask=mask)

@torch.fx.wrap
def triton_silu(input_tensor):
    n_elements = input_tensor.numel()
    BLOCK_SIZE = 1024
    num_programs = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    output = torch.empty_like(input_tensor)
    silu_kernel[(num_programs,)](
        input_tensor, output, n_elements, BLOCK_SIZE
    )
    return output

def replacement_func():
    return triton_silu