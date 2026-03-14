import torch
from torch import device
import triton
import triton.language as tl

def pattern(in_0):
    """
    Match the entire computation graph
    """
    # Create mask tensor
    tmp_0 = torch.zeros((1, 133, 133), device=device(type='cuda', index=0))
    
    # Fill last 5 rows with 1
    tmp_1 = tmp_0[slice(None, None, None), slice(-5, None, None), slice(None, None, None)]
    tmp_2 = tmp_1.fill_(1)
    
    # Fill last 5 columns with 1
    tmp_3 = tmp_0[slice(None, None, None), slice(None, None, None), slice(-5, None, None)]
    tmp_4 = tmp_3.fill_(1)
    
    # Transform input tensor
    tmp_5 = in_0.reshape(1, 19, 7, 19, 7, 96)
    tmp_6 = tmp_5.transpose(2, 3)
    
    # Transform mask tensor
    tmp_7 = tmp_0.reshape(1, 19, 7, 19, 7)
    tmp_8 = tmp_7.transpose(2, 3)
    tmp_9 = tmp_8.reshape(1, 361, 49)
    tmp_10 = tmp_9.unsqueeze(2)
    tmp_11 = tmp_9.unsqueeze(3)
    tmp_12 = tmp_10 - tmp_11
    
    return tmp_12, tmp_6

def replacement_args(in_0):
    return (in_0,)

@triton.jit
def fused_mask_kernel(
    out_ptr,
    M: tl.constexpr,
    N1: tl.constexpr,
    N2: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N1: tl.constexpr,
    BLOCK_N2: tl.constexpr,
):
    """
    Generate the mask tensor directly in final shape (1, 361, 49, 49)
    """
    # Get program IDs
    pid_m = tl.program_id(0)
    pid_n1 = tl.program_id(1)
    pid_n2 = tl.program_id(2)
    
    # Calculate offsets
    m_offs = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    n1_offs = pid_n1 * BLOCK_N1 + tl.arange(0, BLOCK_N1)
    n2_offs = pid_n2 * BLOCK_N2 + tl.arange(0, BLOCK_N2)
    
    # Masks for bounds checking
    m_mask = m_offs < M
    n1_mask = n1_offs < N1
    n2_mask = n2_offs < N2
    
    # Expand dimensions for broadcasting
    m_offs_3d = m_offs[:, None, None]
    n1_offs_3d = n1_offs[None, :, None]
    n2_offs_3d = n2_offs[None, None, :]
    
    # Compute original coordinates
    i = m_offs_3d // 19
    k = m_offs_3d % 19
    
    j_n1 = n1_offs_3d // 7
    l_n1 = n1_offs_3d % 7
    j_n2 = n2_offs_3d // 7
    l_n2 = n2_offs_3d % 7
    
    # Original coordinates in (133, 133) tensor
    orig_row_n1 = i * 7 + j_n1
    orig_col_n1 = k * 7 + l_n1
    orig_row_n2 = i * 7 + j_n2
    orig_col_n2 = k * 7 + l_n2
    
    # Check if in border (last 5 rows or last 5 columns)
    mask_val_n1 = ((orig_row_n1 >= 128) | (orig_col_n1 >= 128)).to(tl.float32)
    mask_val_n2 = ((orig_row_n2 >= 128) | (orig_col_n2 >= 128)).to(tl.float32)
    
    # Compute difference
    result = mask_val_n2 - mask_val_n1
    
    # Create combined mask
    combined_mask = m_mask[:, None, None] & n1_mask[None, :, None] & n2_mask[None, None, :]
    
    # Compute output indices
    out_indices = (m_offs_3d * N1 * N2 + n1_offs_3d * N2 + n2_offs_3d)
    
    # Store result
    tl.store(out_ptr + out_indices, result, mask=combined_mask)

@triton.jit
def fused_reshape_transpose_kernel(
    in_ptr,
    out_ptr,
    N: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fused reshape + transpose kernel
    """
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N
    
    # Decode output linear index to [b, i, k, j, l, c]
    rem = offsets
    c = rem % 96
    rem = rem // 96
    l = rem % 7
    rem = rem // 7
    j = rem % 7
    rem = rem // 7
    k = rem % 19
    rem = rem // 19
    i = rem % 19
    
    # Map to input coordinates
    in_row = i * 7 + j
    in_col = k * 7 + l
    
    # Compute input linear index: (1, 133, 133, 96)
    in_idx = in_row * 133 * 96 + in_col * 96 + c
    
    # Load and store
    val = tl.load(in_ptr + in_idx, mask=mask, other=0.0)
    tl.store(out_ptr + offsets, val, mask=mask)

@torch.fx.wrap
def fused_full_computation(in_0):
    """
    Fused entire computation
    """
    # Generate mask tensor
    mask_out = torch.empty((1, 361, 49, 49), device='cuda', dtype=torch.float32)
    
    M = 361
    N1 = 49
    N2 = 49
    
    BLOCK_M = 32
    BLOCK_N1 = 16
    BLOCK_N2 = 16
    
    grid_mask = (
        triton.cdiv(M, BLOCK_M),
        triton.cdiv(N1, BLOCK_N1),
        triton.cdiv(N2, BLOCK_N2)
    )
    
    fused_mask_kernel[grid_mask](
        mask_out,
        M, N1, N2,
        BLOCK_M, BLOCK_N1, BLOCK_N2
    )
    
    # Transform input tensor
    input_out = torch.empty((1, 19, 19, 7, 7, 96), device=in_0.device, dtype=in_0.dtype)
    
    N = input_out.numel()
    BLOCK_SIZE = 1024
    grid_input = (triton.cdiv(N, BLOCK_SIZE),)
    
    fused_reshape_transpose_kernel[grid_input](
        in_0,
        input_out,
        N,
        BLOCK_SIZE
    )
    
    return mask_out, input_out

def replacement_func():
    return fused_full_computation