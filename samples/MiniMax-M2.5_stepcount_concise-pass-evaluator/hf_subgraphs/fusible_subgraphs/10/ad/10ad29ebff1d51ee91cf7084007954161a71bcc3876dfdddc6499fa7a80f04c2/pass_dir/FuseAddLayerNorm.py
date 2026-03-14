import torch
import triton
import triton.language as tl


def pattern(in_0, in_1, in_2, in_3):
    """
    Pattern: add + layer_norm fusion
    Computes: layer_norm(in_2 + in_3, normalized_shape, weight, bias, eps)
    Returns only tmp_3 (the layer_norm output) since tmp_2 is not in the model return
    """
    tmp_2 = in_2 + in_3
    tmp_3 = torch.nn.functional.layer_norm(tmp_2, (256,), in_1, in_0, 1e-05)
    return tmp_3


def replacement_args(in_0, in_1, in_2, in_3):
    return (in_0, in_1, in_2, in_3)


# Fused kernel: add + layer_norm
# This fuses the first add with layer_norm
@triton.jit
def layer_norm_fused_kernel(
    input_ptr, other_ptr, weight_ptr, bias_ptr,
    output_ptr,
    N: tl.constexpr,
    M: tl.constexpr,
    B: tl.constexpr,
    eps: tl.constexpr,
):
    """
    Fused kernel that computes:
    1. sum = input + other (element-wise)
    2. layer_norm(sum, normalized_shape, weight, bias, eps)
    """
    # Get position
    row_idx = tl.program_id(0)
    batch_idx = row_idx // M
    seq_idx = row_idx % M
    row_offset = batch_idx * M * N + seq_idx * N
    
    offsets = tl.arange(0, 256)
    mask = offsets < N
    
    # Load inputs
    x = tl.load(input_ptr + row_offset + offsets, mask=mask, other=0.0).to(tl.float32)
    y = tl.load(other_ptr + row_offset + offsets, mask=mask, other=0.0).to(tl.float32)
    
    # Compute sum
    sum_vals = x + y
    
    # Layer norm
    mean = tl.sum(sum_vals, axis=0) / N
    var = tl.sum((sum_vals - mean) * (sum_vals - mean), axis=0) / N
    std = tl.sqrt(var + eps)
    normalized = (sum_vals - mean) / std
    
    # Weight and bias
    weight = tl.load(weight_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    bias = tl.load(bias_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    
    # Output
    output = normalized * weight + bias
    tl.store(output_ptr + row_offset + offsets, output, mask=mask)


@torch.fx.wrap
def layer_norm_fused(bias, weight, in_2, in_3, eps=1e-05):
    """
    Fused add + layer_norm kernel
    
    Args:
        bias: [256] - layer norm bias
        weight: [256] - layer norm weight
        in_2: [B, M, N] - first tensor to add
        in_3: [B, M, N] - second tensor to add
        eps: epsilon for numerical stability
    
    Returns:
        layer_norm_output: [B, M, N]
    """
    B, M, N = in_2.shape
    
    # Allocate output
    output = torch.empty_like(in_2)
    
    # Grid: one program per (B * M) row
    grid = (B * M,)
    
    layer_norm_fused_kernel[grid](
        in_2, in_3, weight, bias,
        output,
        N, M, B, eps,
    )
    
    return output


def replacement_func():
    return layer_norm_fused