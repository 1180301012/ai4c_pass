import torch
import triton
import triton.language as tl

def pattern(in_0):
    tmp0 = torch.nn.functional.relu(in_0, inplace=False)
    tmp1 = torch.nn.functional.dropout(tmp0, 0.0, False, False)
    tmp2 = tmp1.flatten(1, -1)
    return (tmp2,)

def replacement_args(in_0):
    return (in_0,)

@triton.jit
def optimized_kernel(
    input_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    block_id = tl.program_id(0)
    block_start = block_id * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    input_vals = tl.load(input_ptr + block_start + offsets, mask=mask, other=0.0)
    relu_vals = tl.maximum(input_vals, 0.0)
    tl.store(output_ptr + block_start + offsets, relu_vals, mask=mask)

@torch.fx.wrap
def kernel_wrapper(input):
    n_elements = input.numel()
    BLOCK_SIZE = 1024
    output = torch.empty(n_elements, dtype=input.dtype, device=input.device)
    optimized_kernel[(n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE,](
        input_ptr=input,
        output_ptr=output,
        n_elements=n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return output
def replacement_func():
    return kernel_wrapper