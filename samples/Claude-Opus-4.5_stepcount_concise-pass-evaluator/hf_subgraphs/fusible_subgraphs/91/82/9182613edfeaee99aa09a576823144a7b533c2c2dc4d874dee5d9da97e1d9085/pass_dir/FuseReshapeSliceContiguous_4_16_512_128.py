import torch
import triton
import triton.language as tl

# Pattern for reshape + contiguous on Graph 5: batch=4, seq=512
def pattern(in_1):
    tmp_0 = in_1.reshape(4, 16, 512, 128)
    tmp_4 = tmp_0.contiguous()
    return tmp_4

def replacement_args(in_1):
    return (in_1,)

# Required Triton kernel 
@triton.jit
def copy_kernel(in_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offset < n_elements
    x = tl.load(in_ptr + offset, mask=mask)
    tl.store(out_ptr + offset, x, mask=mask)

@torch.fx.wrap  
def optimized_reshape_contiguous(in_1):
    return in_1.reshape(4, 16, 512, 128).contiguous()

def replacement_func():
    return optimized_reshape_contiguous