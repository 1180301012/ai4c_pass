import torch
import triton
import triton.language as tl


def pattern(conv_out, norm_weight, norm_bias, normalized_shape):
    """
    Pattern: conv2d output -> flatten(2) -> transpose(1,2) -> layer_norm
    Input: conv_out is [B, C, H, W]
    Output: [B, H*W, C] normalized
    """
    flattened = conv_out.flatten(2)
    transposed = flattened.transpose(1, 2)
    normalized = torch.nn.functional.layer_norm(transposed, normalized_shape, norm_weight, norm_bias, 1e-05)
    return normalized


def replacement_args(conv_out, norm_weight, norm_bias, normalized_shape):
    return (conv_out, norm_weight, norm_bias)


@triton.jit
def fused_flatten_transpose_layernorm_kernel(
    input_ptr,
    weight_ptr,
    bias_ptr,
    output_ptr,
    B: tl.constexpr,
    C: tl.constexpr,
    HW: tl.constexpr,
    eps: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fused kernel for flatten(2) + transpose(1,2) + layer_norm.
    
    Input layout: [B, C, H, W] (flattened to [B, C, HW])
    Output layout: [B, HW, C]
    
    Each program processes one spatial position (one row of output).
    """
    # Get the position we're processing: batch and spatial location
    pid = tl.program_id(0)
    batch_idx = pid // HW
    spatial_idx = pid % HW
    
    # Load weight and bias (shared across all spatial positions)
    # Use BLOCK_SIZE which is a power of 2, with masking
    c_offsets = tl.arange(0, BLOCK_SIZE)
    c_mask = c_offsets < C
    weight = tl.load(weight_ptr + c_offsets, mask=c_mask, other=0.0)
    bias = tl.load(bias_ptr + c_offsets, mask=c_mask, other=0.0)
    
    # Load input data for this spatial position
    # Input is [B, C, HW], we want position [batch_idx, :, spatial_idx]
    input_offsets = batch_idx * C * HW + c_offsets * HW + spatial_idx
    x = tl.load(input_ptr + input_offsets, mask=c_mask, other=0.0)
    
    # Compute mean (only over valid elements)
    mean = tl.sum(x, axis=0) / C
    
    # Compute variance (only over valid elements)
    x_centered = tl.where(c_mask, x - mean, 0.0)  # Zero out masked positions
    variance = tl.sum(x_centered * x_centered, axis=0) / C
    
    # Normalize
    rstd = 1.0 / tl.sqrt(variance + eps)
    x_normalized = x_centered * rstd
    
    # Apply affine transformation
    output = x_normalized * weight + bias
    
    # Store output in transposed layout [B, HW, C]
    output_offsets = batch_idx * HW * C + spatial_idx * C + c_offsets
    tl.store(output_ptr + output_offsets, output, mask=c_mask)


@torch.fx.wrap
def fused_flatten_transpose_layernorm(conv_out, weight, bias):
    """
    Fused implementation of flatten(2) + transpose(1,2) + layer_norm.
    
    Args:
        conv_out: [B, C, H, W] tensor from conv2d
        weight: [C] normalization weight
        bias: [C] normalization bias
    
    Returns:
        [B, H*W, C] normalized tensor
    """
    B, C, H, W = conv_out.shape
    HW = H * W
    
    # Find the next power of 2 >= C for BLOCK_SIZE
    import math
    BLOCK_SIZE = 2 ** math.ceil(math.log2(C))
    
    # Allocate output
    output = torch.empty((B, HW, C), dtype=conv_out.dtype, device=conv_out.device)
    
    # Launch kernel: one program per (batch, spatial_position)
    grid = (B * HW,)
    
    fused_flatten_transpose_layernorm_kernel[grid](
        conv_out,
        weight,
        bias,
        output,
        B=B,
        C=C,
        HW=HW,
        eps=1e-05,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return output


def replacement_func():
    return fused_flatten_transpose_layernorm