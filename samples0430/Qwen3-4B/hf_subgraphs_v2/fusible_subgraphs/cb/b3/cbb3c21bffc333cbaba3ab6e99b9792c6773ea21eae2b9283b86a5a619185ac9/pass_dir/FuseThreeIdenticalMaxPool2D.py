import torch
import triton
import triton.language as tl

def pattern(in_0):
    tmp0 = torch.nn.functional.relu(in_0, inplace=True)
    tmp1 = torch.nn.functional.max_pool2d(tmp0, 5, 1, 2, 1, ceil_mode=False, return_indices=False)
    tmp2 = torch.nn.functional.max_pool2d(tmp0, 5, 1, 2, 1, ceil_mode=False, return_indices=False)
    tmp3 = torch.nn.functional.max_pool2d(tmp0, 5, 1, 2, 1, ceil_mode=False, return_indices=False)
    tmp4 = torch.cat([tmp0, tmp1, tmp2, tmp3], 1)
    return tmp4
def replacement_args(in_0):
    return (in_0,)

@triton.jit
def fused_kernel(
    in_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    in_val = tl.load(in_ptr + offsets, mask=mask, other=0.0)
    relu_val = tl.where(in_val > 0, in_val, 0.0)
    
    pool_val = relu_val
    
    tl.store(out_ptr + offsets, relu_val, mask=mask)
    tl.store(out_ptr + offsets, pool_val, mask=mask)
    tl.store(out_ptr + offsets, pool_val, mask=mask)
    tl.store(out_ptr + offsets, pool_val, mask=mask)

@torch.fx.wrap
def kernel_wrapper(in_0):
    N = in_0.numel()
    BLOCK_SIZE = 1024
    num_programs = (N + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    out = torch.empty_like(in_0)
    
    fused_kernel[(num_programs,)](
        in_ptr=in_0,
        out_ptr=out,
        n_elements=N,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out
def replacement_func():
    return kernel_wrapper