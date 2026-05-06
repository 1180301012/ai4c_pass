import torch
import triton
import triton.language as tl

def pattern(in_0, in_1):
    tmp_0 = torch.softmax(in_1, dim=1)
    tmp_1 = in_0 * tmp_0
    return torch.sum(tmp_1, dim=1)

def replacement_args(in_0, in_1):
    return (in_0, in_1)

@triton.jit
def optimized_kernel(in_0_ptr, in_1_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    block_id = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_SIZE) + block_id * BLOCK_SIZE
    mask = offsets < n_elements
    in_0_0 = tl.load(in_0_ptr + offsets, mask=mask, other=0.0)
    in_0_1 = tl.load(in_0_ptr + offsets + 1, mask=mask, other=0.0)
    a = tl.load(in_1_ptr + offsets, mask=mask, other=0.0)
    b = tl.load(in_1_ptr + offsets + 1, mask=mask, other=0.0)
    a = tl.load(in_1_ptr + offsets, mask=mask, other=0.0)
    b = tl.load(in_1_ptr + offsets + 1, mask=mask, other=0.0)
    num = in_0_0 * a + in_0_1 * b
    denom = a + b
    out = num / denom
    tl.store(output_ptr + offsets, out, mask=mask)

@torch.fx.wrap
def kernel_wrapper(in_0, in_1):
    n_elements = in_0.shape[0]
    output = torch.empty_like(in_0)
    optimized_kernel[(1,)](
        in_0_ptr=in_0,
        in_1_ptr=in_1,
        output_ptr=output,
        n_elements=n_elements,
        BLOCK_SIZE=256
    )
    return output
def replacement_func():
    return kernel_wrapper