import torch
import triton
import triton.language as tl

@triton.jit
def relative_position_mask_kernel(
    seq_len: tl.constexpr,
    offset_mask_ptr,
    base_ptr,
    output_ptr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fused kernel for computing relative position mask.
    
    The original computation:
    1. Creates offset = -sign(diff) * 16 for negative differences
    2. Creates base = clamp(log(|diff|)/ln(10) * 8 + 8, 0, 15) for |diff| < 8
    3. Result = offset + base
    """
    pid = tl.program_id(0)
    num_blocks = (seq_len * seq_len + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    for block_idx in range(num_blocks):
        start_idx = block_idx * BLOCK_SIZE
        offsets = start_idx + tl.arange(0, BLOCK_SIZE)
        mask = offsets < (seq_len * seq_len)
        
        # Compute i, j indices for 2D position
        i_idx = offsets // seq_len
        j_idx = offsets % seq_len
        
        # diff = j - i (j_idx - i_idx)
        diff = j_idx - i_idx
        
        # offset: -sign(diff) * 16 for diff < 0, else 0
        # Simplified: diff < 0 ? -16 : 0
        offset = tl.where(diff < 0, -16, 0)
        
        # base: for |diff| < 8, use |diff|; otherwise clamp(log(|diff|)/ln(10)*8 + 8, 0, 15)
        abs_diff = tl.abs(diff)
        
        # Condition: |diff| < 8
        in_range = abs_diff < 8
        
        # Compute log-based value for |diff| >= 8
        # log(|diff|) / ln(10) * 8 + 8, clamped to [0, 15]
        # Using natural log: log(|diff|) / ln(10) ≈ log10(|diff|)
        # For simplicity, compute log10(abs_diff) * 8 + 8, then clamp
        log_val_float = tl.log(tl.cast(abs_diff, tl.float32)) / 2.302585092994046  # ln(10)
        log_scaled = log_val_float * 8.0 + 8.0
        log_clamped = tl.clamp(log_scaled, 0.0, 15.0)
        log_result = tl.cast(log_clamped, tl.int64)
        
        # Final base: use abs_diff if in_range, else log_clamped value
        base = tl.where(in_range, abs_diff, log_result)
        
        # Final result
        result = offset + base
        
        tl.store(output_ptr + offsets, result, mask=mask)


@torch.fx.wrap
def triton_relative_position_mask(seq_len: int, dtype: torch.dtype):
    """
    Generate the relative position mask using fused Triton kernel.
    Output shape: [seq_len, seq_len]
    """
    N = seq_len * seq_len
    BLOCK_SIZE = 1024
    num_programs = (N + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    output = torch.empty((seq_len, seq_len), dtype=torch.int64, device='cuda')
    
    relative_position_mask_kernel[(num_programs,)](
        seq_len=seq_len,
        offset_mask_ptr=0,
        base_ptr=0,
        output_ptr=output,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return output


def pattern(in_0, in_1, in_2, in_3, in_4, in_5):
    """
    Match the relative position mask computation pattern.
    This includes the arange creation, reshaping, subtraction, and mask computation.
    """
    # Create the arange tensors and reshape
    tmp_10 = torch.arange(in_0.shape[1], dtype=torch.int64)
    tmp_11 = tmp_10[(slice(None, None, None), None)]
    tmp_12 = torch.arange(in_0.shape[1], dtype=torch.int64)
    tmp_13 = tmp_12[(None, slice(None, None, None))]
    
    # Compute difference
    tmp_14 = tmp_13 - tmp_11
    tmp_15 = -tmp_14
    
    # Negative mask offset
    tmp_16 = tmp_15 < 0
    tmp_17 = tmp_16.to(torch.int64)
    tmp_18 = tmp_17 * 16
    tmp_19 = 0 + tmp_18
    
    # Absolute value for log computation
    tmp_20 = torch.abs(tmp_15)
    tmp_21 = tmp_20 < 8
    tmp_22 = tmp_20.float()
    tmp_23 = tmp_22 / 8
    tmp_24 = torch.log(tmp_23)
    tmp_25 = tmp_24 / 2.772588722239781
    tmp_26 = tmp_25 * 8
    tmp_27 = tmp_26.to(torch.int64)
    tmp_28 = 8 + tmp_27
    tmp_29 = torch.full_like(tmp_28, 15)
    tmp_30 = torch.min(tmp_28, tmp_29)
    tmp_31 = torch.where(tmp_21, tmp_20, tmp_30)
    tmp_19 += tmp_31
    tmp_32 = tmp_19
    
    return tmp_32


def replacement_args(in_0, in_1, in_2, in_3, in_4, in_5):
    # Extract sequence length from input tensor shape
    seq_len = in_0.shape[1]
    dtype = in_0.dtype
    return (seq_len, dtype)


def replacement_func():
    return triton_relative_position_mask