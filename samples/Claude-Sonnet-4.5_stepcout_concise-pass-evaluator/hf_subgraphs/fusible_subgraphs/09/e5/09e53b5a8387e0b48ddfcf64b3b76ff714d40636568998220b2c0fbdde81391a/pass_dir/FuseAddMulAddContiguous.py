import torch
import triton
import triton.language as tl
import operator

def pattern(in_modified, in_0, in_3):
    """Match the pattern: (in_modified * in_0) + in_3, then contiguous
    where in_modified is the result after in_2 += in_1"""
    tmp_2 = in_modified * in_0
    tmp_3 = tmp_2 + in_3
    tmp_4 = tmp_3.contiguous()
    return tmp_4

def replacement_args(in_modified, in_0, in_3):
    return (in_modified, in_0, in_3)

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 8192}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=2),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=2),
    ],
    key=['n_elements'],
)
@triton.jit
def fused_mul_add_kernel(
    in_modified_ptr,
    in_3_ptr,
    out_ptr,
    scalar_val,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """Fused kernel: out = (in_modified * scalar_val) + in_3"""
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load inputs
    in_modified = tl.load(in_modified_ptr + offsets, mask=mask, other=0.0)
    in_3 = tl.load(in_3_ptr + offsets, mask=mask, other=0.0)
    
    # Fused computation
    out = in_modified * scalar_val + in_3
    
    # Store result
    tl.store(out_ptr + offsets, out, mask=mask)

@torch.fx.wrap
def fused_mul_add(in_modified, in_0, in_3):
    """Wrapper function for the fused kernel"""
    # Extract scalar value from in_0
    if in_0.numel() == 1:
        scalar_val = in_0.item()
    else:
        scalar_val = in_0.flatten()[0].item()
    
    n_elements = in_modified.numel()
    
    # Allocate output tensor
    out = torch.empty_like(in_modified)
    
    # Launch kernel with grid
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    
    fused_mul_add_kernel[grid](
        in_modified,
        in_3,
        out,
        scalar_val,
        n_elements,
    )
    
    return out

def replacement_func():
    return fused_mul_add