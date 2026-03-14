import torch
import triton
import triton.language as tl

def pattern(tmp_0):
    tmp_1 = tmp_0[slice(None, None, None), slice(-5, None, None), slice(None, None, None)]
    tmp_2 = tmp_1.fill_(1)
    tmp_3 = tmp_0[slice(None, None, None), slice(None, None, None), slice(-5, None, None)]
    tmp_4 = tmp_3.fill_(1)
    return (tmp_2, tmp_4)

def replacement_args(tmp_0):
    return (tmp_0,)

@triton.jit
def fill_ones_kernel(
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    tl.store(out_ptr + offsets, 1.0, mask=mask)

@torch.fx.wrap
def optimized_slice_and_fill(tmp_0):
    h, w = tmp_0.shape[1], tmp_0.shape[2]
    
    # Create output tensors filled with ones directly
    out1 = torch.empty((1, 5, w), dtype=tmp_0.dtype, device=tmp_0.device)
    out2 = torch.empty((1, h, 5), dtype=tmp_0.dtype, device=tmp_0.device)
    
    # Fill them with ones using Triton for better performance
    BLOCK_SIZE = 1024
    
    # Fill out1
    n_elements1 = out1.numel()
    num_programs1 = (n_elements1 + BLOCK_SIZE - 1) // BLOCK_SIZE
    fill_ones_kernel[(num_programs1,)](out1, n_elements1, BLOCK_SIZE=BLOCK_SIZE)
    
    # Fill out2  
    n_elements2 = out2.numel()
    num_programs2 = (n_elements2 + BLOCK_SIZE - 1) // BLOCK_SIZE
    fill_ones_kernel[(num_programs2,)](out2, n_elements2, BLOCK_SIZE=BLOCK_SIZE)
    
    return out1, out2

def replacement_func():
    return optimized_slice_and_fill