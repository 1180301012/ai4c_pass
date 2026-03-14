import torch
import triton
import triton.language as tl


# Kernel using the same algorithm as PyTorch's LayerNorm
@triton.jit(do_not_specialize=['eps'])
def fused_layernorm_kernel(
    input_ptr, add_ptr, weight_ptr, bias_ptr, output_ptr,
    N, M,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    # Get row ID
    row_id = tl.program_id(0)
    row_offset = row_id * N
    
    # Offsets
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < N
    
    # Load and add
    x = tl.load(input_ptr + row_offset + offsets, mask=mask, other=0.0) + \
        tl.load(add_ptr + row_offset + offsets, mask=mask, other=0.0)
    
    # Compute mean - exactly like PyTorch: sum(x) / N
    mean = tl.sum(x, axis=0) / N
    
    # Variance: mean((x - mean)^2) - same as original computation
    x_centered = x - mean
    variance = tl.sum(x_centered * x_centered, axis=0) / N
    
    # Compute std - add eps first for numerical stability
    std = tl.sqrt(variance + eps)
    
    # Normalize: (x - mean) / std
    normalized = x_centered / std
    
    # Load weight and bias
    w = tl.load(weight_ptr + offsets, mask=mask, other=0.0)
    b = tl.load(bias_ptr + offsets, mask=mask, other=0.0)
    
    # Output: weight * normalized + bias
    output = normalized * w + b
    tl.store(output_ptr + row_offset + offsets, output, mask=mask)


def _fused_layernorm_impl(in_3, in_2, weight, bias, eps):
    shape = in_3.shape
    N = shape[-1]
    M = in_3.numel() // N
    
    in_3_flat = in_3.view(M, N)
    in_2_flat = in_2.view(M, N)
    out = torch.empty_like(in_3_flat)
    
    grid = (M,)
    BLOCK_SIZE = 1024
    fused_layernorm_kernel[grid](in_3_flat, in_2_flat, weight, bias, out, N, M, eps, BLOCK_SIZE)
    
    return out.view(shape)


@torch.fx.wrap
def fused_layernorm(in_3, in_2, weight, bias, eps=1e-07):
    return _fused_layernorm_impl(in_3, in_2, weight, bias, eps)


def pattern(in_0, in_1, in_2, in_3):
    """
    Match the LayerNorm computation pattern:
    1. Add: tmp_3 = in_3 + in_2
    2. To float: tmp_4 = tmp_3.float()
    3. Mean: tmp_5 = tmp_4.mean(-1, keepdim=True)
    4. Subtract mean: tmp_6 = tmp_4 - tmp_5
    5. Square: tmp_7 = tmp_6.pow(2)
    6. Mean of squares: tmp_8 = tmp_7.mean(-1, keepdim=True)
    7. Re-subtract mean: tmp_9 = tmp_4 - tmp_5
    8. Add epsilon: tmp_10 = tmp_8 + 1e-07
    9. Sqrt: tmp_11 = torch.sqrt(tmp_10)
    10. Divide: tmp_12 = tmp_9 / tmp_11
    11. To float32: tmp_13 = tmp_12.to(torch.float32)
    12. Multiply by weight: tmp_14 = in_1 * tmp_13
    13. Add bias: tmp_15 = tmp_14 + in_0
    
    Returns tmp_15 (the final output)
    """
    # Step 1: Add in_2 to in_3
    tmp_3 = in_3 + in_2
    
    # Step 2: Convert to float
    tmp_4 = tmp_3.float()
    
    # Step 3: Compute mean along last dimension
    tmp_5 = tmp_4.mean(-1, keepdim=True)
    
    # Step 4: Subtract mean (for variance calculation)
    tmp_6 = tmp_4 - tmp_5
    
    # Step 5: Square
    tmp_7 = tmp_6.pow(2)
    
    # Step 6: Compute mean of squares (variance)
    tmp_8 = tmp_7.mean(-1, keepdim=True)
    
    # Step 7: Re-subtract mean (for normalization) - squeeze to match shapes
    tmp_9 = tmp_4 - tmp_5
    
    # Step 8: Add epsilon
    tmp_10 = tmp_8 + 1e-07
    
    # Step 9: Square root
    tmp_11 = torch.sqrt(tmp_10)
    
    # Step 10: Normalize
    tmp_12 = tmp_9 / tmp_11
    
    # Step 11: Convert to float32 (redundant but part of pattern)
    tmp_13 = tmp_12.to(torch.float32)
    
    # Step 12: Multiply by weight
    tmp_14 = in_1 * tmp_13
    
    # Step 13: Add bias
    tmp_15 = tmp_14 + in_0
    
    return tmp_15


def replacement_args(in_0, in_1, in_2, in_3):
    """
    Extract arguments for the replacement function.
    in_0: bias
    in_1: weight
    in_2: add tensor
    in_3: input tensor
    """
    return (in_0, in_1, in_2, in_3)


def replacement_func():
    """
    Return the fused LayerNorm implementation.
    """
    return fused_layernorm