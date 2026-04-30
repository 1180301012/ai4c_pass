import torch
import triton
import triton.language as tl


# Fixed BLOCK_SIZE constant for this kernel (power of 2 for 768 elements)
BLOCK_SIZE_CONST = tl.constexpr(1024)


@triton.jit
def fused_add_scale_layernorm_kernel(
    in_2_ptr,
    in_3_ptr,
    weight_ptr,
    bias_ptr,
    output_ptr,
    n_elements: tl.constexpr,
    eps: tl.constexpr,
):
    """
    Fused kernel: (in_2 + in_3) / 2 -> LayerNorm
    Uses BLOCK_SIZE_CONST = 1024 for optimal performance with 768 elements
    Single program to ensure correct mean/variance computation
    """
    # Get program ID (single program for this 1D reduction)
    pid = tl.program_id(0)
    
    # Calculate offsets for loading using constexpr BLOCK_SIZE
    offsets = pid * BLOCK_SIZE_CONST + tl.arange(0, BLOCK_SIZE_CONST)
    mask = offsets < n_elements
    
    # Load inputs with float32 accumulation for precision
    in_2 = tl.load(in_2_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    in_3 = tl.load(in_3_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    
    # Add and scale by 0.5
    x = (in_2 + in_3) * 0.5
    
    # Compute mean using reduction across all elements
    mean = tl.sum(x, axis=0)
    mean = mean / n_elements
    
    # Compute variance (using two-pass method for numerical stability)
    x_centered = x - mean
    x_centered_sq = x_centered * x_centered
    var = tl.sum(x_centered_sq, axis=0) / n_elements
    
    # Standard deviation with epsilon for numerical stability
    std = tl.sqrt(var + eps)
    
    # Normalize
    normalized = x_centered / std
    
    # Load weight and bias for affine transformation
    weight = tl.load(weight_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    bias = tl.load(bias_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    
    # Apply affine transformation: y = normalized * weight + bias
    output = normalized * weight + bias
    
    # Store result
    tl.store(output_ptr + offsets, output, mask=mask)


@torch.fx.wrap
def fused_add_scale_layernorm(in_0, in_1, in_2, in_3):
    """
    Fused add + scale + layer_norm implementation.
    Performs: layer_norm((in_2 + in_3) / 2, (768,), in_1, in_0, 1e-12)
    Args order matches pattern: in_0=bias, in_1=weight, in_2=input_a, in_3=input_b
    """
    weight = in_1
    bias = in_0
    in_2_tensor = in_2
    in_3_tensor = in_3
    eps = 1e-12
    
    n_elements = in_2_tensor.numel()
    
    # Create output tensor with same dtype as input
    output = torch.empty_like(in_2_tensor)
    
    # Launch kernel with 1 program (single program ensures correct global mean/variance)
    grid = (1,)
    
    fused_add_scale_layernorm_kernel[grid](
        in_2_ptr=in_2_tensor,
        in_3_ptr=in_3_tensor,
        weight_ptr=weight,
        bias_ptr=bias,
        output_ptr=output,
        n_elements=n_elements,
        eps=eps,
    )
    
    return output


def pattern(in_0, in_1, in_2, in_3):
    """
    Pattern: add + div + layer_norm
    This matches the computation:
        tmp_2 = in_2 + in_3
        tmp_3 = tmp_2 / 2
        tmp_4 = layer_norm(tmp_3, (768,), in_1, in_0, 1e-12)
    """
    tmp_2 = in_2 + in_3
    tmp_3 = tmp_2 / 2
    tmp_4 = torch.nn.functional.layer_norm(tmp_3, (768,), in_1, in_0, 1e-12)
    return tmp_4


def replacement_args(in_0, in_1, in_2, in_3):
    """
    Extract arguments needed for the fused kernel.
    """
    return (in_0, in_1, in_2, in_3)


def replacement_func():
    """
    Returns the module-level fused kernel function.
    """
    return fused_add_scale_layernorm