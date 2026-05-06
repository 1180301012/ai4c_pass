import torch
import triton
import triton.language as tl

def pattern(tmp_4):
    return torch.nn.functional.softmax(tmp_4, dim=2)

def replacement_args(tmp_4):
    return (tmp_4,)

@triton.jit
def softmax_kernel(
    input_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr
):
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    exp_x = tl.exp(x)
    sum_exp = tl.sum(exp_x)
    softmax = exp_x / sum_exp
    tl.store(output_ptr + offsets, softmax, mask=mask)

@torch.fx.wrap
def softmax_replace(tmp_4):
    B, C, D = tmp_4.shape
    n_elements = B * C * D
    output = torch.empty_like(tmp_4)
    softmax_kernel[(tl.cdiv(n_elements, 256),)](
        input_ptr=tmp_4,
        output_ptr=output,
        n_elements=n_elements,
        BLOCK_SIZE=256
    )
    return output

def replacement_func():
    return softmax_replace