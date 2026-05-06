import torch
import triton
import triton.language as tl

def pattern(in_0, in_1):
    in_1 = in_1 + in_0
    x = in_1.float()
    softmax_results = torch.nn.functional.softmax(x, dim=-1)
    y = softmax_results.type_as(in_1)
    output = y * 0.9
    return (output,)

def replacement_args(in_0, in_1):
    return (in_0, in_1)

@triton.jit
def softmax_triton_kernel(x_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = (block_start + offsets) < n_elements

    x = tl.load(x_ptr + block_start + offsets, mask=mask, other=-1e10)
    max_val = tl.max(x)
    x_exp = tl.exp(x - max_val)
    sum_exp = tl.sum(x_exp, axis=-1)
    softmax = x_exp / sum_exp

    tl.store(output_ptr + block_start + offsets, softmax, mask=mask)

@torch.fx.wrap
def kernel_wrapper(x, y):
    sum_tensor = x + y
    output_tensor = torch.empty_like(sum_tensor, dtype=torch.float32)
    n_elements = sum_tensor.numel()
    num_blocks = (n_elements + 1024 - 1) // 1024

    softmax_triton_kernel[(num_blocks,)](
        x_ptr=sum_tensor.data_ptr(),
        output_ptr=output_tensor.data_ptr(),
        n_elements=n_elements,
        BLOCK_SIZE=1024,
    )
    output_tensor = output_tensor.type_as(y)
    output_tensor = output_tensor * 0.9
    return (output_tensor,)

def replacement_func():
    return kernel_wrapper