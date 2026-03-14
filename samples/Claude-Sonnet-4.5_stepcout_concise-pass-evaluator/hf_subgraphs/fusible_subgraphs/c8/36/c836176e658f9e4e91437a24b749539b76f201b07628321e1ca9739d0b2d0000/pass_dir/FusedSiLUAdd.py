import torch
import triton
import triton.language as tl

def pattern(a, b):
    """Match simple add operation"""
    return a + b

def replacement_args(a, b):
    return (a, b)

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 256}),
        triton.Config({'BLOCK_SIZE': 512}),
        triton.Config({'BLOCK_SIZE': 1024}),
        triton.Config({'BLOCK_SIZE': 2048}),
    ],
    key=['n_elements'],
)
@triton.jit
def add_kernel(
    a_ptr,
    b_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    a = tl.load(a_ptr + offsets, mask=mask, other=0.0)
    b = tl.load(b_ptr + offsets, mask=mask, other=0.0)
    out = a + b
    
    tl.store(out_ptr + offsets, out, mask=mask)

@torch.fx.wrap
def triton_add(a, b):
    a_contig = a.contiguous()
    b_contig = b.contiguous()
    n_elements = a_contig.numel()
    out = torch.empty_like(a_contig)
    
    grid = lambda meta: ((n_elements + meta['BLOCK_SIZE'] - 1) // meta['BLOCK_SIZE'],)
    
    add_kernel[grid](
        a_ptr=a_contig,
        b_ptr=b_contig,
        out_ptr=out,
        n_elements=n_elements,
    )
    
    return out

def replacement_func():
    return triton_add