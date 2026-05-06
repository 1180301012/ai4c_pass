import torch
import triton
import triton.language as tl

def pattern(x):
    relu_x = torch.nn.functional.relu(x, inplace=True)
    sigmoid_x = torch.sigmoid(relu_x)
    return sigmoid_x
def replacement_args(x):
    return (x,)

@triton.jit
def fused_relu_sigmoid_kernel(
    input_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    block_id = tl.program_id(0)
    offsets = block_id * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    input_vals = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    relu_vals = tl.where(input_vals > 0, input_vals, 0.0)
    sigmoid_vals = 1.0 / (1.0 + tl.exp(-relu_vals))
    tl.store(output_ptr + offsets, sigmoid_vals, mask=mask)

@torch.fx.wrap
def kernel_wrapper(x):
    N = x.numel()
    BLOCK_SIZE = 1024
    num_programs = (N + BLOCK_SIZE - 1) // BLOCK_SIZE
    output = torch.empty_like(x)
    fused_relu_sigmoid_kernel[(num_programs,)](
        input_ptr=x,
        output_ptr=output,
        n_elements=N,
        BLOCK_SIZE=BLOCK_SIZE
    )
    return output
def replacement_func():
    return kernel_wrapper