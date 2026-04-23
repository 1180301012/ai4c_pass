import torch
import triton
import triton.language as tl


@triton.jit
def fused_layernorm_kernel(
    x_ptr,
    residual_ptr,
    weight_ptr,
    bias_ptr,
    output_ptr,
    num_rows,
    num_features,
    EPS,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fused LayerNorm kernel.
    Uses BLOCK_SIZE=1024 (power of 2 >= 768) for tl.arange.
    """
    pid = tl.program_id(0)
    
    # Row boundaries
    row_start = pid * num_features
    row_end = row_start + num_features
    
    # Offsets within block (BLOCK_SIZE = 1024 >= num_features = 768)
    feat_idx = tl.arange(0, BLOCK_SIZE)
    
    # Feature mask: only first num_features elements are valid
    feat_mask = feat_idx < num_features
    
    # Load x and residual, add them
    x_offset = row_start + feat_idx
    x_mask = x_offset < row_end
    x = tl.load(x_ptr + x_offset, mask=x_mask, other=0.0)
    residual = tl.load(residual_ptr + x_offset, mask=x_mask, other=0.0)
    x = x + residual
    
    # Convert to float32
    x = x.to(tl.float32)
    
    # Zero out invalid positions for reduction
    x_valid = tl.where(feat_mask, x, 0.0)
    
    # Compute mean
    sum_val = tl.sum(x_valid)
    mean_val = sum_val / num_features
    
    # Center
    centered = x - mean_val
    
    # Zero out invalid positions for variance computation
    centered_valid = tl.where(feat_mask, centered, 0.0)
    
    # Compute variance
    var_sum = tl.sum(centered_valid * centered_valid)
    var_val = var_sum / num_features
    
    # Std
    std_val = tl.sqrt(var_val + EPS)
    
    # Normalize
    normalized = centered / std_val
    
    # Zero out invalid positions for output
    normalized_valid = tl.where(feat_mask, normalized, 0.0)
    
    # Load weight and bias
    weight = tl.load(weight_ptr + feat_idx)
    bias = tl.load(bias_ptr + feat_idx)
    
    # Apply affine transform
    result = normalized_valid * weight + bias
    
    # Store (only valid positions)
    tl.store(output_ptr + x_offset, result, mask=x_mask)


@torch.fx.wrap
def triton_fused_layernorm_autotuned(x, residual, weight, bias):
    """
    Wrapper function for the fused LayerNorm kernel.
    
    Args:
        x: hidden_states tensor with shape [..., num_features]
        residual: tensor to add with same shape as x
        weight: LayerNorm weight with shape [num_features]
        bias: LayerNorm bias with shape [num_features]
    """
    # Calculate dimensions
    num_features = x.shape[-1]
    total_elements = x.numel()
    num_rows = total_elements // num_features
    
    # Allocate output as float32 to match eager behavior
    output = torch.empty(x.shape, dtype=torch.float32, device=x.device)
    
    # Use BLOCK_SIZE = 1024 (smallest power of 2 >= num_features = 768)
    BLOCK_SIZE = 1024
    
    # Launch kernel: one program per row
    grid = (num_rows,)
    fused_layernorm_kernel[grid](
        x,
        residual,
        weight,
        bias,
        output,
        num_rows,
        num_features,
        1e-7,
        BLOCK_SIZE,
    )
    
    return output


def pattern(in_0, in_1, in_2, in_3):
    """
    Match the LayerNorm computation pattern:
    1. Add hidden_states + residual
    2. Convert to float
    3. Compute mean
    4. Center the data
    5. Compute variance (mean of squared deviations)
    6. Normalize by std
    7. Scale by weight
    8. Add bias
    """
    # Step 1: Add hidden_states + residual
    tmp_3 = in_3 + in_2

    # Step 2: Convert to float
    tmp_4 = tmp_3.float()

    # Step 3: Compute mean over last dimension
    tmp_5 = tmp_4.mean(-1, keepdim=True)

    # Step 4: Subtract mean (center)
    tmp_6 = tmp_4 - tmp_5

    # Step 5: Square
    tmp_7 = tmp_6.pow(2)

    # Step 6: Compute variance (mean of squared)
    tmp_8 = tmp_7.mean(-1, keepdim=True)

    # Step 7: Subtract mean again (needed for normalization)
    tmp_9 = tmp_4 - tmp_5

    # Step 8: Add epsilon
    tmp_10 = tmp_8 + 1e-07

    # Step 9: Square root (std)
    tmp_11 = torch.sqrt(tmp_10)

    # Step 10: Normalize
    tmp_12 = tmp_9 / tmp_11

    # Step 11: Convert to float32 (may be no-op depending on input)
    tmp_13 = tmp_12.to(torch.float32)

    # Step 12: Scale by weight
    tmp_14 = in_1 * tmp_13

    # Step 13: Add bias
    tmp_15 = tmp_14 + in_0

    return tmp_15


def replacement_args(in_0, in_1, in_2, in_3):
    """
    Extract arguments needed for the fused kernel.
    """
    return (in_0, in_1, in_2, in_3)


def replacement_func():
    """
    Return the optimized function.
    """
    return triton_fused_layernorm_autotuned