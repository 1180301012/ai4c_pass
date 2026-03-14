import torch
import triton
import triton.language as tl


def pattern(tmp_0, tmp_1, tmp_2):
    """Match: layer_norm + slice pattern"""
    tmp_7 = torch.nn.functional.layer_norm(tmp_2, (512,), tmp_1, tmp_0, 1e-06)
    tmp_8 = tmp_7[slice(None, None, None), 0]
    return tmp_8


def replacement_args(tmp_0, tmp_1, tmp_2):
    return (tmp_0, tmp_1, tmp_2)


@triton.jit
def fused_layernorm_slice_kernel(
    in_ptr,
    weight_ptr, bias_ptr,
    out_ptr,
    D,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fused kernel for layer_norm + slice (only row 0)
    Only processes row 0 since we only need the output for [:, 0]
    """
    # Only process row 0
    row_offset = 0
    
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < D
    
    # Step 1: Load data (row 0)
    data = tl.load(in_ptr + row_offset + offsets, mask=mask, other=0.0)
    
    # Step 2: Compute layer norm
    # Mean
    masked_val = tl.where(mask, data, 0.0)
    mean = tl.sum(masked_val) / D
    
    # Variance
    centered = data - mean
    masked_sq = tl.where(mask, centered * centered, 0.0)
    variance = tl.sum(masked_sq) / D
    
    # Normalize
    rstd = 1.0 / tl.sqrt(variance + eps)
    normalized = centered * rstd
    
    # Apply affine transform
    weight = tl.load(weight_ptr + offsets, mask=mask, other=1.0)
    bias = tl.load(bias_ptr + offsets, mask=mask, other=0.0)
    output = normalized * weight + bias
    
    # Step 3: Store output
    tl.store(out_ptr + offsets, output, mask=mask)


@torch.fx.wrap
def fused_layernorm_slice(tmp_0, tmp_1, tmp_2):
    """
    Fused implementation of layer_norm + slice
    Args:
        tmp_0: bias [512]
        tmp_1: weight [512]
        tmp_2: input tensor [1, 145, 512]
    Returns:
        tmp_8: [1, 512] - layer_norm result sliced at [:, 0]
    """
    B, N, D = tmp_2.shape
    
    # Allocate output
    tmp_8 = torch.empty((B, D), dtype=tmp_2.dtype, device=tmp_2.device)
    
    # Launch kernel: one program for row 0
    BLOCK_SIZE = triton.next_power_of_2(D)
    grid = (1,)
    
    fused_layernorm_slice_kernel[grid](
        tmp_2,
        tmp_1, tmp_0,
        tmp_8,
        D,
        1e-06,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return tmp_8


def replacement_func():
    return fused_layernorm_slice