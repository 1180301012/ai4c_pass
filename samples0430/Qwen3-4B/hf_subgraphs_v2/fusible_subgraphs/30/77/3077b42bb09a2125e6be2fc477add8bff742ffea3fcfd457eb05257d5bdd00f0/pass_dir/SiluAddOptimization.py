import torch
import triton
import triton.language as tl

def pattern(in_0, in_1):
    tmp_0 = torch.nn.functional.silu(in_1, inplace=True)
    tmp_1 = tmp_0 + in_0
    return tmp_1
def replacement_args(in_0, in_1):
    return (in_0, in_1)

@triton.jit
def silu_add_kernel(
    in_0_ptr,
    in_1_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    i = tl.program_id(0)
    block_start = i * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = (block_start + offsets) < n_elements
    
    in_0 = tl.load(in_0_ptr + block_start + offsets, mask=mask, other=0.0)
    in_1 = tl.load(in_1_ptr + block_start + offsets, mask=mask, other=0.0)
    
    exp_val = tl.exp(-in_1)
    sigmoid_val = 1.0 / (1.0 + exp_val)
    silu_val = in_1 * sigmoid_val
    
    result = silu_val + in_0
    
    tl.store(out_ptr + block_start + offsets, result, mask=mask)

@torch.fx.wrap
def kernel_wrapper(in_0, in_1):
    n_elements = in_0.numel()
    BLOCK_SIZE = 256
    num_programs = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    out = torch.empty_like(in_0)
    
    silu_add_kernel[(num_programs,)](
        in_0_ptr=in_0,
        in_1_ptr=in_1,
        out_ptr=out,
        n_elements=n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return out
def replacement_func():
    return kernel_wrapper