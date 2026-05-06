import torch
import triton
import triton.language as tl

def pattern(in0, in1, in2):
    conv = torch.conv2d(in2, in1, in0, (1, 1), (0, 0), (1, 1), 1)
    reshaped = conv.view(1, 1, -1)
    return torch.softmax(reshaped, dim=-1)

def replacement_args(in0, in1, in2):
    return (in0, in1, in2)

@triton.jit
def softmax_kernel(input_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    block_id = tl.program_id(0)
    block_start = block_id * BLOCK_SIZE
    block_end = min(block_start + BLOCK_SIZE, n_elements)
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < (block_end - block_start)

    x = tl.load(input_ptr + block_start + offsets, mask=mask, other=0.0)
    max_val = tl.max(x)
    exp_x = tl.exp(x - max_val)
    sum_exp = tl.sum(exp_x)
    softmax = exp_x / sum_exp

    tl.store(output_ptr + block_start + offsets, softmax, mask=mask)

@torch.fx.wrap
def kernel_wrapper(in0, in1, in2):
    conv = torch.conv2d(in2, in1, in0, (1, 1), (0, 0), (1, 1), 1)
    n_elements = conv.numel()
    output = torch.empty(n_elements, dtype=conv.dtype, device=conv.device)
    softmax_kernel[tl.cdiv(n_elements, 256)](\n        input_ptr=conv,
        output_ptr=output,
        n_elements=n_elements,
        BLOCK_SIZE=256,
    )
    return output

def replacement_func():
    return kernel_wrapper