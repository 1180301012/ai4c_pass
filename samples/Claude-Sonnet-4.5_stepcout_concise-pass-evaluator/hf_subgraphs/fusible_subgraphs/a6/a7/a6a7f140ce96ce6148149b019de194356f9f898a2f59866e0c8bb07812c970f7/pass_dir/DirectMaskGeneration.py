import torch
from torch import device
import triton
import triton.language as tl

def pattern():
    """
    Match the mask creation and transformation to final subtraction result
    """
    tmp_0 = torch.zeros((1, 133, 133), device=device(type='cuda', index=0))
    tmp_1 = tmp_0[slice(None, None, None), slice(-5, None, None), slice(None, None, None)]
    tmp_2 = tmp_1.fill_(1)
    tmp_3 = tmp_0[slice(None, None, None), slice(None, None, None), slice(-5, None, None)]
    tmp_4 = tmp_3.fill_(1)
    tmp_7 = tmp_0.reshape(1, 19, 7, 19, 7)
    tmp_8 = tmp_7.transpose(2, 3)
    tmp_9 = tmp_8.reshape(1, 361, 49)
    tmp_10 = tmp_9.unsqueeze(2)
    tmp_11 = tmp_9.unsqueeze(3)
    tmp_12 = tmp_10 - tmp_11
    return tmp_12

def replacement_args():
    return ()

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=8),
    ],
    key=['N_ELEMENTS'],
)
@triton.jit
def generate_mask_kernel(
    out_ptr,
    N_ELEMENTS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Directly generate the final mask output (1, 361, 49, 49)
    """
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N_ELEMENTS
    
    # Decode output index (1, 361, 49, 49) -> (batch=0, m, n1, n2)
    rem = offsets
    n2 = rem % 49
    rem = rem // 49
    n1 = rem % 49
    rem = rem // 49
    m = rem  # m in [0, 360]
    
    # Map back to original (133, 133) coordinates
    # m = i*19 + k where i,k in [0,19)
    # n = j*7 + l where j,l in [0,7)
    # Original: [i*7+j, k*7+l]
    
    i = m // 19
    k = m % 19
    j_n1 = n1 // 7
    l_n1 = n1 % 7
    j_n2 = n2 // 7
    l_n2 = n2 % 7
    
    # Original coordinates in (133, 133)
    orig_row_n1 = i * 7 + j_n1
    orig_col_n1 = k * 7 + l_n1
    orig_row_n2 = i * 7 + j_n2
    orig_col_n2 = k * 7 + l_n2
    
    # Check if in border (last 5 rows: 128-132 or last 5 cols: 128-132)
    in_border_n1 = ((orig_row_n1 >= 128) | (orig_col_n1 >= 128))
    in_border_n2 = ((orig_row_n2 >= 128) | (orig_col_n2 >= 128))
    
    # Convert to float (1.0 if in border, 0.0 otherwise)
    val_n1 = tl.where(in_border_n1, 1.0, 0.0)
    val_n2 = tl.where(in_border_n2, 1.0, 0.0)
    
    # Compute difference
    result = val_n2 - val_n1
    
    # Store
    tl.store(out_ptr + offsets, result, mask=mask)

@torch.fx.wrap
def generate_mask_direct():
    """
    Generate mask directly in final form
    """
    out = torch.empty((1, 361, 49, 49), device='cuda', dtype=torch.float32)
    
    N_ELEMENTS = 361 * 49 * 49
    
    grid = lambda meta: (triton.cdiv(N_ELEMENTS, meta['BLOCK_SIZE']),)
    
    generate_mask_kernel[grid](
        out,
        N_ELEMENTS,
    )
    
    return out

def replacement_func():
    return generate_mask_direct