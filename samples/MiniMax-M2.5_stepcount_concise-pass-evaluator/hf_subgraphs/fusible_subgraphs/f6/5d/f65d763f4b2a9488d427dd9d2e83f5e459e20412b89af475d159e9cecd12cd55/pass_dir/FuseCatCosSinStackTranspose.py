import torch
import triton
import triton.language as tl


# Pattern matching function - matches the exact computation from model.py
def pattern(in_0, in_1, in_2):
    """
    Define the computation pattern to match:
    1. cat(in_0, in_2) along dim=-1 -> tmp_0 [64, 128]
    2. cos(in_1) -> tmp_1 [64, 64]
    3. sin(in_1) -> tmp_2 [64, 64]
    4. cat(tmp_1, tmp_2) along dim=-1 -> tmp_3 [64, 128]
    5. stack(tmp_0, tmp_3) along dim=-1 -> tmp_4 [64, 128, 2]
    6. transpose tmp_4 -> tmp_5 [64, 2, 128]
    """
    tmp_0 = torch.cat((in_0, in_2), dim=-1)
    tmp_1 = in_1.cos()
    tmp_2 = in_1.sin()
    tmp_3 = torch.cat((tmp_1, tmp_2), dim=-1)
    tmp_4 = torch.stack((tmp_0, tmp_3), dim=-1)
    tmp_5 = tmp_4.transpose(-1, -2)
    return tmp_5


# Extract arguments needed for replacement
def replacement_args(in_0, in_1, in_2):
    return (in_0, in_1, in_2)


# Triton kernel using 2D grid - one program per row
@triton.jit
def fused_kernel(
    in_0_ptr, in_1_ptr, in_2_ptr, out_ptr,
    M, N,
    stride_in_0_0, stride_in_0_1,
    stride_in_1_0, stride_in_1_1,
    stride_in_2_0, stride_in_2_1,
    stride_out_0, stride_out_1, stride_out_2,
):
    """
    Use 2D grid: (M, 2) = (64, 2)
    Each program processes one row of one stack slice.
    """
    # Grid dimensions: program_id(0) = row (0-63), program_id(1) = stack_idx (0-1)
    row_idx = tl.program_id(0)
    stack_idx = tl.program_id(1)
    
    # Use fixed constexpr values - input is always [64, 64]
    K: tl.constexpr = 128  # 2 * N where N=64
    N_half: tl.constexpr = 64  # N = 64
    
    # Process all columns for this row and stack_idx
    col_offsets = tl.arange(0, K)
    mask = col_offsets < K
    
    if stack_idx == 0:
        # Output[stack_idx=0] = tmp_0 = cat(in_0, in_2)
        # For col < N: from in_0, for col >= N: from in_2
        # Load in_0[row, col]
        in_0_idx = row_idx * stride_in_0_0 + col_offsets * stride_in_0_1
        val_in_0 = tl.load(in_0_ptr + in_0_idx, mask=mask, other=0.0)
        
        # Load in_2[row, col-N] 
        in_2_col = col_offsets - N_half
        in_2_idx = row_idx * stride_in_2_0 + in_2_col * stride_in_2_1
        val_in_2 = tl.load(in_2_ptr + in_2_idx, mask=mask, other=0.0)
        
        # Select based on column
        result = tl.where(col_offsets < N_half, val_in_0, val_in_2)
    else:
        # Output[stack_idx=1] = tmp_3 = cat(cos(in_1), sin(in_1))
        # For col < N: from cos(in_1), for col >= N: from sin(in_1)
        
        # Load in_1[row, col]
        in_1_idx = row_idx * stride_in_1_0 + col_offsets * stride_in_1_1
        val_in_1 = tl.load(in_1_ptr + in_1_idx, mask=mask, other=0.0)
        
        # Compute cos and sin
        val_cos = tl.cos(val_in_1)
        val_sin = tl.sin(val_in_1)
        
        # Select based on column
        result = tl.where(col_offsets < N_half, val_cos, val_sin)
    
    # Store to output[row, stack_idx, col]
    # output shape: [M, 2, K]
    out_idx = row_idx * stride_out_0 + stack_idx * stride_out_1 + col_offsets * stride_out_2
    tl.store(out_ptr + out_idx, result, mask=mask)


@torch.fx.wrap
def fused_kernel_wrapper(in_0, in_1, in_2):
    """Launch the fused Triton kernel with 2D grid."""
    M, N = in_0.shape  # [64, 64]
    K = 2 * N  # 128
    
    # Output shape: [M, 2, K] = [64, 2, 128]
    output = torch.empty((M, 2, K), dtype=torch.float32, device=in_0.device)
    
    # 2D grid: (M, 2) = (64, 2)
    grid = (M, 2)
    
    fused_kernel[grid](
        in_0, in_1, in_2, output,
        M, N,
        in_0.stride(0), in_0.stride(1),
        in_1.stride(0), in_1.stride(1),
        in_2.stride(0), in_2.stride(1),
        output.stride(0), output.stride(1), output.stride(2),
    )
    
    return output


def replacement_func():
    return fused_kernel_wrapper