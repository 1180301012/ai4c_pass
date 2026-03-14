import torch
import triton
import triton.language as tl

# Simple copy kernel with larger block for better throughput
@triton.jit
def cat_copy_first(src_ptr, dst_ptr, n_elements, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_elements
    x = tl.load(src_ptr + offs, mask=mask)
    tl.store(dst_ptr + offs, x, mask=mask)

@triton.jit  
def cat_copy_second(src_ptr, dst_ptr, offset, n_elements, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_elements
    x = tl.load(src_ptr + offs, mask=mask)
    tl.store(dst_ptr + offset + offs, x, mask=mask)

@torch.fx.wrap
def optimized_cat(in_0, in_1):
    n1 = in_1.numel()
    n0 = in_0.numel()
    
    out_rows = in_1.shape[0] + in_0.shape[0]
    out_cols = in_1.shape[1]
    out = torch.empty((out_rows, out_cols), dtype=in_1.dtype, device=in_1.device)
    
    # Use optimal block size
    BLOCK = 1024
    
    # Copy first tensor (in_1)
    grid1 = ((n1 + BLOCK - 1) // BLOCK,)
    cat_copy_first[grid1](in_1, out, n1, BLOCK=BLOCK)
    
    # Copy second tensor (in_0) with offset
    grid0 = ((n0 + BLOCK - 1) // BLOCK,)
    cat_copy_second[grid0](in_0, out, n1, n0, BLOCK=BLOCK)
    
    return out

def pattern(in_0, in_1):
    tmp_0 = torch.cat([in_1, in_0])
    return tmp_0

def replacement_args(in_0, in_1):
    return (in_0, in_1)

def replacement_func():
    return optimized_cat