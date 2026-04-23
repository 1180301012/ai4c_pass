import torch
import triton
import triton.language as tl

# Try matching using torch.relu (which FX should trace properly)
def pattern(in_0):
    tmp_0 = torch.relu(in_0)
    tmp_1 = tmp_0.flatten(1, -1)
    return (tmp_1,)

def replacement_args(in_0):
    return (in_0,)

@triton.jit
def relu_flatten_kernel(
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
    out = tl.where(x > 0, x, 0.0)
    tl.store(output_ptr + offsets, out, mask=mask)

@torch.fx.wrap
def relu_flatten_kernel_wrapper(in_0):
    n_elements = in_0.numel()
    BLOCK_SIZE = 1024
    num_programs = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE

    batch_size = in_0.shape[0]
    flattened_size = in_0.numel() // batch_size
    output_shape = (batch_size, flattened_size)

    output = torch.empty(output_shape, dtype=in_0.dtype, device=in_0.device)

    relu_flatten_kernel[(num_programs,)](
        input_ptr=in_0,
        output_ptr=output,
        n_elements=n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    return output

def replacement_func():
    return relu_flatten_kernel_wrapper