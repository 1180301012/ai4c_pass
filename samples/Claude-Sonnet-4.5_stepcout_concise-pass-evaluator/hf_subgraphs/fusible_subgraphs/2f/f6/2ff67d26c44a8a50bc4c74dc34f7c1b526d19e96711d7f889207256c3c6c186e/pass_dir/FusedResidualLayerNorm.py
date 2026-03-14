import torch
import triton
import triton.language as tl


def pattern(in_0, in_1, in_2, in_3):
    """
    Pattern matching for residual + LayerNorm + affine transformation.
    This matches the exact computation in model.py.
    """
    tmp_3 = in_3 + in_2
    tmp_4 = tmp_3.float()
    tmp_5 = tmp_4.mean(-1, keepdim=True)
    tmp_6 = tmp_4 - tmp_5
    tmp_7 = tmp_6.pow(2)
    tmp_8 = tmp_7.mean(-1, keepdim=True)
    tmp_9 = tmp_4 - tmp_5
    tmp_10 = tmp_8 + 1e-07
    tmp_11 = torch.sqrt(tmp_10)
    tmp_12 = tmp_9 / tmp_11
    tmp_13 = tmp_12.to(torch.float32)
    tmp_14 = in_1 * tmp_13
    tmp_15 = tmp_14 + in_0
    return (tmp_15,)


def replacement_args(in_0, in_1, in_2, in_3):
    """Extract arguments for replacement function."""
    return (in_0, in_1, in_2, in_3)


@triton.jit
def fused_residual_layernorm_kernel(
    in_2_ptr,
    in_3_ptr,
    weight_ptr,
    bias_ptr,
    out_ptr,
    N,  # Hidden dimension (768)
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fused kernel for residual + LayerNorm + affine transformation.
    Each program processes one row (one sequence position).
    Optimized version with better memory access.
    """
    # Get row index
    row_idx = tl.program_id(0)
    
    # Calculate row start pointer
    row_start = row_idx * N
    
    # Load offsets
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < N
    
    # Load in_2 and in_3, compute residual
    in_2 = tl.load(in_2_ptr + row_start + offsets, mask=mask, other=0.0).to(tl.float32)
    in_3 = tl.load(in_3_ptr + row_start + offsets, mask=mask, other=0.0).to(tl.float32)
    residual = in_2 + in_3
    
    # Compute mean (use where to handle masking properly)
    residual_sum = tl.sum(tl.where(mask, residual, 0.0), axis=0)
    mean = residual_sum / N
    
    # Compute variance
    residual_centered = tl.where(mask, residual - mean, 0.0)
    variance = tl.sum(residual_centered * residual_centered, axis=0) / N
    rstd = 1.0 / tl.sqrt(variance + eps)
    
    # Load weight and bias
    weight = tl.load(weight_ptr + offsets, mask=mask, other=1.0).to(tl.float32)
    bias = tl.load(bias_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    
    # Normalize and apply affine transformation
    normalized = (residual - mean) * rstd
    out = normalized * weight + bias
    
    # Store result
    tl.store(out_ptr + row_start + offsets, out, mask=mask)


@torch.fx.wrap
def fused_residual_layernorm(bias, weight, in_2, in_3):
    """
    Wrapper function for the fused residual + LayerNorm kernel.
    
    Args:
        bias: [N] - LayerNorm bias
        weight: [N] - LayerNorm weight
        in_2: [B, S, N] - First input tensor
        in_3: [B, S, N] - Second input tensor (residual)
    
    Returns:
        out: [B, S, N] - Output after residual + LayerNorm + affine
    """
    # Get shape info
    *batch_shape, N = in_3.shape
    total_rows = in_3.numel() // N
    
    # Flatten to 2D for processing
    in_2_flat = in_2.reshape(-1, N).contiguous()
    in_3_flat = in_3.reshape(-1, N).contiguous()
    
    # Allocate output
    out = torch.empty_like(in_3_flat)
    
    # Use fixed BLOCK_SIZE of 1024 (next power of 2 >= 768)
    BLOCK_SIZE = 1024
    
    # Launch kernel - one program per row
    grid = (total_rows,)
    
    fused_residual_layernorm_kernel[grid](
        in_2_flat,
        in_3_flat,
        weight,
        bias,
        out,
        N,
        1e-07,
        BLOCK_SIZE,
    )
    
    # Reshape back to original shape
    out = out.reshape(in_3.shape)
    
    return out


def replacement_func():
    """Return the replacement function."""
    return fused_residual_layernorm