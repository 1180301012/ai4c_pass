import torch
import triton
import triton.language as tl

def pattern(in_2, in_1, in_0):
    conv2d = torch.conv2d(in_2, in_1, in_0, (1, 1), (0, 0), (1, 1), 1)
    permuted = conv2d.permute(0, 2, 3, 1)
    reshaped = permuted.reshape(in_2.shape[0], -1, in_1.shape[1])
    return torch.nn.functional.sigmoid(reshaped)

def replacement_args(in_2, in_1, in_0):
    return (in_2, in_1, in_0)

@triton.jit
def optimized_kernel(
    input_ptr,
    output_ptr,
    n_elements,
    num_rows,
    k,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offset = pid * BLOCK_SIZE
    input_vals = tl.load(input_ptr + offset, mask=tl.arange(0, BLOCK_SIZE) < n_elements, other=0.0)
    output_vals = tl.sigmoid(input_vals)
    tl.store(output_ptr + offset, output_vals, mask=tl.arange(0, BLOCK_SIZE) < n_elements)

@torch.fx.wrap
def optimized_func(in_2, in_1, in_0):
    n, c, h, w = in_2.shape
    num_elements = n * h * w
    k = in_1.shape[1]
    output = torch.empty((n, num_elements // n, k), device=in_2.device, dtype=in_2.dtype)
    BLOCK_SIZE = 1024
    num_programs = (num_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    optimized_kernel[(num_programs,)](
        input_ptr=in_2,
        output_ptr=output,
        n_elements=num_elements,
        num_rows=n,
        k=k,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return output
def replacement_func():
    return optimized_func