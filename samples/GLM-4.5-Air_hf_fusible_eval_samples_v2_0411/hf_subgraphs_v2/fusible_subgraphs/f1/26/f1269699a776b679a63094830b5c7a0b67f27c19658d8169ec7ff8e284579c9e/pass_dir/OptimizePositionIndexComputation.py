def pattern(in_0):
    """
    Pattern matching for the complex position index computation sequence.
    This matches the computation from tmp_10 to tmp_32 in the model.
    The sequence length is derived from the input shape.
    """
    # Extract sequence length from input shape
    seq_len = in_0.shape[1]
    
    tmp_10 = torch.arange(seq_len, dtype=torch.int64)
    tmp_11 = tmp_10[(slice(None, None, None), None)]
    tmp_12 = torch.arange(seq_len, dtype=torch.int64)
    tmp_13 = tmp_12[(None, slice(None, None, None))]
    tmp_14 = tmp_13 - tmp_11
    tmp_15 = -tmp_14
    tmp_16 = tmp_15 < 0
    tmp_17 = tmp_16.to(torch.int64)
    tmp_18 = tmp_17 * 16
    tmp_19 = 0 + tmp_18
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

def replacement_args(in_0):
    return (in_0,)

# Define optimized Triton kernel for position index computation
@triton.jit
def position_index_kernel(
    seq_len_ptr,
    output_ptr,
    BLOCK_SIZE: tl.constexpr,
):
    # Calculate global position
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < (seq_len * seq_len)
    
    # Load sequence length
    seq_len = tl.load(seq_len_ptr)
    
    # Compute grid coordinates
    i = offsets // seq_len  # row index
    j = offsets % seq_len   # column index
    
    # Compute position difference j - i
    diff = j - i
    
    # Compute negative difference (-diff)
    neg_diff = -diff
    
    # Check if original difference was positive (neg_diff < 0)
    is_positive_original = neg_diff < 0
    
    # Initialize output with scaled positive differences
    # is_positive_original * 16 gives weight of 16 for positions where j > i
    output = is_positive_original * 16
    
    # Compute absolute difference
    abs_diff = tl.abs(neg_diff)
    
    # Check if absolute difference is less than 8
    is_small_diff = abs_diff < 8
    
    # For small absolute differences (< 8), use absolute difference directly
    # For larger differences, use logarithmic transformation:
    # log(abs_diff/8) / ln(16) * 8 + 8, then clamp to max 15
    
    # Compute value for larger differences using logarithmic transformation
    # Handle division by zero when abs_diff == 0
    abs_diff_safe = tl.where(abs_diff == 0, 1.0, abs_diff.astype(tl.float32))
    normalized = abs_diff_safe / 8.0
    logged = tl.log(normalized)
    scaled = logged / 2.772588722239781  # divide by ln(16)
    quantized = (scaled * 8.0).to(tl.int64)
    larger_value = 8 + quantized
    clamped = tl.minimum(larger_value, 15)
    
    # Combine results: use abs_diff for small differences, clamped value for larger
    large_diff_value = tl.where(is_small_diff, abs_diff, clamped)
    
    # Add to the base output
    output += large_diff_value
    
    # Store result
    tl.store(output_ptr + offsets, output, mask=mask)

@torch.fx.wrap
def optimized_position_index_computation(in_0):
    """Wrapper function to launch the optimized Triton kernel"""
    # Extract sequence length from input
    seq_len = in_0.shape[1]
    device = in_0.device
    
    # For CPU devices, use PyTorch implementation to avoid device mismatch
    if device.type == 'cpu':
        tmp_10 = torch.arange(seq_len, dtype=torch.int64)
        tmp_11 = tmp_10[(slice(None, None, None), None)]
        tmp_12 = torch.arange(seq_len, dtype=torch.int64)
        tmp_13 = tmp_12[(None, slice(None, None, None))]
        tmp_14 = tmp_13 - tmp_11
        tmp_15 = -tmp_14
        tmp_16 = tmp_15 < 0
        tmp_17 = tmp_16.to(torch.int64)
        tmp_18 = tmp_17 * 16
        tmp_19 = 0 + tmp_18
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
        return tmp_19
    
    # For CUDA devices, use optimized Triton kernel
    total_elements = seq_len * seq_len
    BLOCK_SIZE = 1024
    num_programs = (total_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Create output tensor on the same device as input
    output = torch.empty((seq_len, seq_len), dtype=torch.int64, device=device)
    
    # Flatten the output for the kernel
    output_flat = output.flatten()
    
    # Launch kernel
    position_index_kernel[(num_programs,)](
        seq_len_ptr=seq_len,
        output_ptr=output_flat,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return output

def replacement_func():
    return optimized_position_index_computation