import torch
import triton
import triton.language as tl

# Pattern matching: sum + div fusion for normalization
def pattern(in_1):
    """
    Match sum(dim=2, keepdim=True) followed by division.
    in_1 shape: [1, 2, 8, 8]
    """
    tmp_0 = in_1.sum(dim=2, keepdim=True)
    tmp_1 = in_1 / tmp_0
    return tmp_1

def replacement_args(in_1):
    return (in_1,)

@triton.jit
def triton_normalize_keepdim_kernel(
    in_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    # Input shape: [1, 2, 8, 8] -> 128 elements total
    # dim=2 means sum over 8 elements in the third dimension
    # For shape [1, 2, 8, 8] with row-major layout:
    # stride[0] = 2*8*8 = 128, stride[1] = 8*8 = 64, stride[2] = 8, stride[3] = 1
    # In linear offset: offset = d0*128 + d1*64 + d2*8 + d3
    # For dim 2 sum, elements at same (d0, d1, d3) but different d2 differ by 8 in linear offset
    
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Calculate indices for output [1, 2, 8, 8]
    d3 = offsets % 8          # dim 3: index 0-7
    rem = offsets // 8
    d2 = rem % 8              # dim 2: index 0-7 (not directly used but extracted)
    rem = rem // 8
    d1 = rem % 2              # dim 1: index 0-1
    d0 = rem // 2             # dim 0: index 0
    
    # For sum(dim=2), we sum over all d2 values for same (d0, d1, d3)
    # Load all 8 elements for this group and sum them using reduction
    base_offset = d0 * 128 + d1 * 64 + d3
    
    x0 = tl.load(in_ptr + base_offset + 0 * 8)
    x1 = tl.load(in_ptr + base_offset + 1 * 8)
    x2 = tl.load(in_ptr + base_offset + 2 * 8)
    x3 = tl.load(in_ptr + base_offset + 3 * 8)
    x4 = tl.load(in_ptr + base_offset + 4 * 8)
    x5 = tl.load(in_ptr + base_offset + 5 * 8)
    x6 = tl.load(in_ptr + base_offset + 6 * 8)
    x7 = tl.load(in_ptr + base_offset + 7 * 8)
    
    sum_val = x0 + x1 + x2 + x3 + x4 + x5 + x6 + x7
    
    # Load current element and normalize
    x = tl.load(in_ptr + offsets, mask=mask, other=0.0)
    out = x / (sum_val + 1e-8)
    
    # Store result
    tl.store(out_ptr + offsets, out, mask=mask)

@torch.fx.wrap
def triton_normalize_keepdim(in_1):
    """
    Optimized normalization: sum(dim=2, keepdim=True) followed by division.
    Input shape: [1, 2, 8, 8]
    Output shape: [1, 2, 8, 8]
    """
    assert in_1.shape == (1, 2, 8, 8), f"Expected shape [1, 2, 8, 8], got {in_1.shape}"
    
    N = in_1.numel()  # 128
    BLOCK_SIZE = 128
    num_programs = 1  # Single program for this small tensor
    
    out = torch.empty_like(in_1)
    
    triton_normalize_keepdim_kernel[(num_programs,)](
        in_ptr=in_1,
        out_ptr=out,
        n_elements=N,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out

def replacement_func():
    return triton_normalize_keepdim