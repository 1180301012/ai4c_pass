import torch
from torch import device
import triton
import triton.language as tl

def pattern():
    """
    Match the entire mask creation and transformation pipeline
    """
    # Create zeros tensor
    tmp_0 = torch.zeros((1, 133, 133), device=device(type='cuda', index=0))
    
    # Fill last 5 rows with 1
    tmp_1 = tmp_0[slice(None, None, None), slice(-5, None, None), slice(None, None, None)]
    tmp_2 = tmp_1.fill_(1)
    
    # Fill last 5 columns with 1
    tmp_3 = tmp_0[slice(None, None, None), slice(None, None, None), slice(-5, None, None)]
    tmp_4 = tmp_3.fill_(1)
    
    # Reshape and transform
    tmp_7 = tmp_0.reshape(1, 19, 7, 19, 7)
    tmp_8 = tmp_7.transpose(2, 3)
    tmp_9 = tmp_8.reshape(1, 361, 49)
    tmp_10 = tmp_9.unsqueeze(2)
    tmp_11 = tmp_9.unsqueeze(3)
    tmp_12 = tmp_10 - tmp_11
    
    return tmp_12

def replacement_args():
    return ()

@triton.jit
def fused_mask_kernel(
    out_ptr,
    BATCH: tl.constexpr,
    M: tl.constexpr,
    N1: tl.constexpr,
    N2: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N1: tl.constexpr,
    BLOCK_N2: tl.constexpr,
):
    """
    Generate the mask tensor directly in final shape (1, 361, 49, 49)
    
    The mask is computed by:
    1. Creating (1, 133, 133) with last 5 rows and columns set to 1
    2. Reshaping to (1, 19, 7, 19, 7) then transpose(2,3) to (1, 19, 19, 7, 7)
    3. Reshaping to (1, 361, 49)
    4. Computing tmp_10[b,m,1,n2] - tmp_11[b,m,n1,1]
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
    
    # For each output position [b=0, m, n1, n2], compute:
    # result = mask_flat[m, n2] - mask_flat[m, n1]
    # where mask_flat is the (361, 49) tensor after transformations
    
    # To compute mask_flat[m, n]:
    # m ranges from 0 to 360 (361 total)
    # n ranges from 0 to 48 (49 total)
    # 
    # After inverse transformations:
    # m = i*19 + k where i in [0,19), k in [0,19)
    # n = j*7 + l where j in [0,7), l in [0,7)
    # This maps to position [i, k, j, l] in (19, 19, 7, 7) tensor (after transpose)
    # Which originally was [i, j, k, l] in (19, 7, 19, 7) tensor
    # Which maps to [i*7+j, k*7+l] in original (133, 133) tensor
    
    # Expand dimensions for broadcasting
    m_offs_3d = m_offs[:, None, None]  # (BLOCK_M, 1, 1)
    n1_offs_3d = n1_offs[None, :, None]  # (1, BLOCK_N1, 1)
    n2_offs_3d = n2_offs[None, None, :]  # (1, 1, BLOCK_N2)
    
    # Compute mask_flat values for n1 and n2
    # For position (m, n):
    i = m_offs_3d // 19
    k = m_offs_3d % 19
    
    j_n1 = n1_offs_3d // 7
    l_n1 = n1_offs_3d % 7
    j_n2 = n2_offs_3d // 7
    l_n2 = n2_offs_3d % 7
    
    # Original coordinates
    orig_row_n1 = i * 7 + j_n1  # Row in original (133, 133)
    orig_col_n1 = k * 7 + l_n1  # Col in original (133, 133)
    orig_row_n2 = i * 7 + j_n2
    orig_col_n2 = k * 7 + l_n2
    
    # Check if in border (last 5 rows or last 5 columns)
    # Row indices 128-132 or column indices 128-132
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

@torch.fx.wrap
def fused_mask_creation():
    """
    Create the mask tensor directly in final shape
    """
    # Output shape: (1, 361, 49, 49)
    out = torch.empty((1, 361, 49, 49), device='cuda', dtype=torch.float32)
    
    BATCH = 1
    M = 361
    N1 = 49
    N2 = 49
    
    BLOCK_M = 32
    BLOCK_N1 = 16
    BLOCK_N2 = 16
    
    grid = (
        triton.cdiv(M, BLOCK_M),
        triton.cdiv(N1, BLOCK_N1),
        triton.cdiv(N2, BLOCK_N2)
    )
    
    fused_mask_kernel[grid](
        out,
        BATCH, M, N1, N2,
        BLOCK_M, BLOCK_N1, BLOCK_N2
    )
    
    return out

def replacement_func():
    return fused_mask_creation