import torch
import triton
import triton.language as tl

def pattern(relu_input):
    """Pattern to match ReLU operation"""
    relu_output = torch.nn.functional.relu(relu_input, inplace=True)
    return relu_output

def replacement_args(relu_input):
    return (relu_input,)

@triton.jit
def relu_kernel(
    input_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load input
    x = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    
    # Apply ReLU
    out = tl.maximum(x, 0.0)
    
    # Store output
    tl.store(output_ptr + offsets, out, mask=mask)

@torch.fx.wrap
def triton_relu(input_tensor):
    N = input_tensor.numel()
    BLOCK_SIZE = 1024
    num_programs = (N + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    output = torch.empty_like(input_tensor)
    
    relu_kernel[(num_programs,)](
        input_ptr=input_tensor,
        output_ptr=output,
        n_elements=N,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return output

def replacement_func():
    return triton_relu