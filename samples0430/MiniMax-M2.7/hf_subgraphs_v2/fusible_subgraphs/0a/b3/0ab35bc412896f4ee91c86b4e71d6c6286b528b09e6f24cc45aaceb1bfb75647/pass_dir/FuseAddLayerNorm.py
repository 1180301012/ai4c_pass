import torch
import triton
import triton.language as tl


@triton.jit
def fused_add_layer_norm_kernel(
    x_ptr, y_ptr, output_ptr,
    weight_ptr, bias_ptr,
    n_elements,
    n_norm: tl.constexpr,
    eps: tl.constexpr,
    BLOCK_SIZE: tl.constexpr
):
    """
    Fused addition + Layer Normalization kernel.
    
    Each program handles one normalization group.
    BLOCK_SIZE is passed as constexpr for arange().
    
    Optimizations:
    - Single program per normalization group for efficient reduction
    - Inline computation of mean and variance
    - Minimal register pressure
    """
    # Get program ID - each block handles one normalization group
    pid = tl.program_id(0)
    
    # Calculate starting offset for this group
    group_offset = pid * n_norm
    
    # Offsets for this block - all elements in the group
    offsets = group_offset + tl.arange(0, BLOCK_SIZE)
    
    # Load x and y (element-wise add) - convert to float32 for precision
    x = tl.load(x_ptr + offsets).to(tl.float32)
    y = tl.load(y_ptr + offsets).to(tl.float32)
    hidden = x + y
    
    # Compute mean using reduction
    sum_hidden = tl.sum(hidden)
    mean = sum_hidden / n_norm
    
    # Compute variance
    diff = hidden - mean
    sum_diff_sq = tl.sum(diff * diff)
    var = sum_diff_sq / n_norm
    
    # Compute normalized output
    std = tl.sqrt(var + eps)
    normalized = diff / std
    
    # Load weight and bias for affine transform
    w = tl.load(weight_ptr + tl.arange(0, BLOCK_SIZE)).to(tl.float32)
    b = tl.load(bias_ptr + tl.arange(0, BLOCK_SIZE)).to(tl.float32)
    
    # Apply weight and bias
    output = normalized * w + b
    
    # Store result
    tl.store(output_ptr + offsets, output)


@torch.fx.wrap
def fused_add_layer_norm(x: torch.Tensor, y: torch.Tensor, 
                         weight: torch.Tensor, bias: torch.Tensor,
                         eps: float) -> torch.Tensor:
    """
    Fused addition + Layer Normalization using optimized kernel.
    
    x, y: Inputs of shape [*, normalized_shape] where normalized_shape is the last dimension
    weight, bias: Layer norm parameters of shape [normalized_shape]
    eps: Epsilon for numerical stability
    """
    # Get dimensions
    assert x.shape == y.shape, "x and y must have same shape"
    last_dim = x.shape[-1]
    n_elements = x.numel()
    n_groups = n_elements // last_dim
    
    # Output tensor
    output = torch.empty_like(x)
    
    # BLOCK_SIZE equals n_norm (must be constexpr for arange)
    BLOCK_SIZE = last_dim
    
    # Grid: one program per normalization group
    grid = (n_groups,)
    
    # Launch kernel
    fused_add_layer_norm_kernel[grid](
        x, y, output,
        weight, bias,
        n_elements,
        last_dim,
        eps,
        BLOCK_SIZE
    )
    
    return output


# Module-level wrapper function for replacement_func
def _fused_add_layer_norm_wrapper(in_0, in_1, in_2, in_3):
    """
    Wrapper that matches the pattern argument order.
    
    Pattern: torch.nn.functional.layer_norm(tmp_2, (128,), in_1, in_0, 1e-05)
    Where: tmp_2 = in_3 + in_2
    
    Args:
        in_0: bias for layer_norm (shape [128])
        in_1: weight for layer_norm (shape [128])
        in_2: first tensor to add (shape [..., 128])
        in_3: second tensor to add (shape [..., 128])
    
    Returns:
        Layer normalized result of in_2 + in_3
    """
    bias = in_0
    weight = in_1
    x = in_2
    y = in_3
    return fused_add_layer_norm(x, y, weight, bias, 1e-05)


def pattern(in_0, in_1, in_2, in_3):
    """
    Pattern to match: element-wise addition followed by layer normalization.
    
    model.py has:
        tmp_2 = in_3 + in_2
        tmp_3 = torch.nn.functional.layer_norm(tmp_2, (128,), in_1, in_0, 1e-05)
    """
    tmp_2 = in_3 + in_2
    tmp_3 = torch.nn.functional.layer_norm(tmp_2, (128,), in_1, in_0, 1e-05)
    return tmp_3


def replacement_args(in_0, in_1, in_2, in_3):
    """
    Extract arguments for the replacement function.
    
    in_0: bias (shape [128])
    in_1: weight (shape [128])
    in_2: first tensor to add (shape [1, 4, 128])
    in_3: second tensor to add (shape [1, 4, 128])
    """
    return (in_0, in_1, in_2, in_3)


def replacement_func():
    """
    Return the fused function.
    """
    return _fused_add_layer_norm_wrapper