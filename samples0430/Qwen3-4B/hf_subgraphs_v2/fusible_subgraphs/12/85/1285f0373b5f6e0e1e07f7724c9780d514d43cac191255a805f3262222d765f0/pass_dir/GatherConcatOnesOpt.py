import torch
import triton
import triton.language as tl

def pattern(a, b, c):
    tmp_1 = a.gather(1, c)
    tmp_9 = torch.cat([tmp_1, b], dim=1)
    size_tmp1 = tmp_1.size(1)
    total_size = 128 + size_tmp1
    tmp_11 = torch.ones(total_size, dtype=torch.float32, device=a.device)
    return (tmp_9, tmp_11)

def replacement_args(a, b, c):
    return (a, b, c)

@triton.jit
def gather_concat_kernel(a_ptr, b_ptr, c_ptr, out1_ptr, out2_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    row = tl.program_id(0)
    offset = row * BLOCK_SIZE
    mask = tl.arange(0, BLOCK_SIZE) < n_elements
    x = tl.load(a_ptr + offset, mask=mask, other=0.0)
    y = tl.load(b_ptr + offset, mask=mask, other=0.0)
    out = x + y
    tl.store(out1_ptr + offset, out, mask=mask)
    
    ones = tl.full((n_elements,), 1.0, dtype=tl.float32)
    tl.store(out2_ptr, ones, mask=tl.arange(0, n_elements) < n_elements)

@torch.fx.wrap
def kernel_wrapper(a, b, c):
    n_elements = c.shape[0]
    out1 = torch.empty((a.shape[0], 128 + n_elements), dtype=torch.float32, device=a.device)
    out2 = torch.empty(n_elements + 128, dtype=torch.float32, device=a.device)
    
    gather_concat_kernel[(1,)]( 
        a_ptr=a, 
        b_ptr=b, 
        c_ptr=c, 
        out1_ptr=out1, 
        out2_ptr=out2, 
        n_elements=n_elements, 
        BLOCK_SIZE=256, 
    )
    
    return (out1, out2)

def replacement_func():
    return kernel_wrapper