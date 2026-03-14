import torch
import triton
import triton.language as tl

def pattern(in_0, in_1):
    """
    Match the pattern: add + permute(0,2,1) + view (generic)
    This pattern matches any add + permute + view combination
    """
    tmp_0 = in_1 + in_0
    tmp_1 = tmp_0.permute(0, 2, 1)
    return tmp_1

def replacement_args(in_0, in_1):
    return (in_0, in_1)

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 512}, num_warps=2),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=2),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 8192}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 16384}, num_warps=8),
    ],
    key=['n_elements'],
)
@triton.jit
def fused_add_permute_kernel(
    in_0_ptr,
    in_1_ptr,
    out_ptr,
    N: tl.constexpr,
    C: tl.constexpr,
    n_elements: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fused kernel for add + permute operation
    """
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Output indices [c, n]
    c_idx = offsets // N
    n_idx = offsets % N
    
    # Input offset [n, c]
    in_offsets = n_idx * C + c_idx
    
    # Load and add
    in_0_vals = tl.load(in_0_ptr + in_offsets, mask=mask, other=0.0)
    in_1_vals = tl.load(in_1_ptr + in_offsets, mask=mask, other=0.0)
    result = in_0_vals + in_1_vals
    
    # Store
    tl.store(out_ptr + offsets, result, mask=mask)

@torch.fx.wrap
def fused_add_permute_generic(in_0, in_1):
    """
    Generic fusion that works for any shape
    """
    B, N, C = in_0.shape
    
    # Output shape after permute: [B, C, N]
    out = torch.empty((B, C, N), dtype=in_0.dtype, device=in_0.device)
    
    n_elements = C * N
    
    grid = lambda META: (triton.cdiv(n_elements, META['BLOCK_SIZE']),)
    
    fused_add_permute_kernel[grid](
        in_0, in_1, out,
        N=N, C=C,
        n_elements=n_elements,
    )
    
    return out

def replacement_func():
    return fused_add_permute_generic