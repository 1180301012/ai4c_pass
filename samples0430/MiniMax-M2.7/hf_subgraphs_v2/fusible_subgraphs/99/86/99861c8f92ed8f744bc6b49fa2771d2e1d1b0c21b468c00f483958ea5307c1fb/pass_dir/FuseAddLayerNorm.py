import torch
import triton
import triton.language as tl


# ==============================================================================
# Triton Kernels for Fused Add + LayerNorm
# ==============================================================================

@triton.jit
def layer_norm_kernel(
    sum_ptr,
    weight_ptr,
    bias_ptr,
    out_norm_ptr,
    n_elements,
    normalized_size: tl.constexpr,
    n_rows: tl.constexpr,
    eps: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Compute layer normalization for the sum tensor.
    This kernel computes mean and variance per row, then normalizes.
    One program per row for efficient reduction.
    """
    pid = tl.program_id(0)
    
    # Each program handles one row
    row_id = pid
    if row_id >= n_rows:
        return
    
    # Compute offsets for this row
    row_start = row_id * normalized_size
    offsets = row_start + tl.arange(0, normalized_size)
    
    # Load values for this row
    vals = tl.load(sum_ptr + offsets)
    
    # Compute mean using reduction
    mean = tl.sum(vals, axis=0) / normalized_size
    
    # Compute variance
    var = tl.sum((vals - mean) * (vals - mean), axis=0) / normalized_size
    std = tl.sqrt(var + eps)
    
    # Normalize
    normed = (vals - mean) / std
    
    # Load weight and bias
    w = tl.load(weight_ptr + tl.arange(0, normalized_size))
    b = tl.load(bias_ptr + tl.arange(0, normalized_size))
    
    # Apply affine transform
    out = normed * w + b
    
    # Store result
    tl.store(out_norm_ptr + offsets, out)


@torch.fx.wrap
def _layer_norm_kernel_wrapper(sum_tensor, weight, bias, normalized_shape, eps, out_norm, n_rows):
    """Internal wrapper for layer_norm kernel."""
    n_elements = sum_tensor.numel()
    normalized_size = normalized_shape[0] if isinstance(normalized_shape, tuple) else normalized_shape
    
    BLOCK_SIZE_LN = 1024
    grid_size_ln = n_rows
    
    layer_norm_kernel[(grid_size_ln,)](
        sum_ptr=sum_tensor,
        weight_ptr=weight,
        bias_ptr=bias,
        out_norm_ptr=out_norm,
        n_elements=n_elements,
        normalized_size=normalized_size,
        n_rows=n_rows,
        eps=eps,
        BLOCK_SIZE=BLOCK_SIZE_LN,
    )


@torch.fx.wrap
def _add_layer_norm_fused(x1, x2, weight, bias, normalized_shape, eps, return_sum_first):
    """
    Fused add + layer_norm implementation.
    
    Args:
        x1: First tensor to add [*, normalized_shape]
        x2: Second tensor to add [*, normalized_shape]
        weight: Layer norm weight [normalized_shape]
        bias: Layer norm bias [normalized_shape]
        normalized_shape: Shape to normalize over
        eps: Epsilon for numerical stability
        return_sum_first: If True, return (sum, norm), else (norm, sum)
    
    Returns:
        Tuple of (sum, norm) or (norm, sum) depending on return_sum_first
    """
    n_elements = x1.numel()
    normalized_size = normalized_shape[0] if isinstance(normalized_shape, tuple) else normalized_shape
    n_rows = n_elements // normalized_size
    
    # Compute sum (this is simple enough that we'll do it inline)
    # but we use empty_like for the output allocation
    out_sum = torch.empty_like(x1)
    
    # Do the addition in-place into out_sum
    out_sum.copy_(x1)
    out_sum.add_(x2)
    
    # Allocate output for normalized
    out_norm = torch.empty_like(x1)
    
    # Compute layer_norm on the sum
    _layer_norm_kernel_wrapper(out_sum, weight, bias, normalized_shape, eps, out_norm, n_rows)
    
    if return_sum_first:
        return out_sum, out_norm
    else:
        return out_norm, out_sum


# ==============================================================================
# Module-level shared replacement function
# ==============================================================================

def _dispatch_wrapper(in_0, in_1, in_2, in_3, route=""):
    """
    Module-level dispatch wrapper that handles both return orderings.
    The route is passed as the last argument to differentiate.
    """
    # Route string determines the return order
    # "sum_first" -> Aniemore models: return (sum, norm)
    # "norm_first" -> Hubert models: return (norm, sum)
    if route == "sum_first":
        return _add_layer_norm_fused(in_2, in_3, in_1, in_0, (1024,), 1e-05, True)
    elif route == "norm_first":
        return _add_layer_norm_fused(in_2, in_3, in_1, in_0, (1024,), 1e-05, False)
    else:
        # Default: sum_first
        return _add_layer_norm_fused(in_2, in_3, in_1, in_0, (1024,), 1e-05, True)


# ==============================================================================
# Pass 1: Match Aniemore pattern (return sum, norm)
# ==============================================================================

def pattern(in_0, in_1, in_2, in_3):
    """
    Match pattern: add + layer_norm for Aniemore models (return (sum, norm)).
    """
    tmp_2 = in_2 + in_3
    tmp_4 = torch.nn.functional.layer_norm(tmp_2, (1024,), in_1, in_0, 1e-05)
    return tmp_2, tmp_4


def replacement_args(in_0, in_1, in_2, in_3):
    """Extract arguments with route string for sum_first pattern."""
    return (in_0, in_1, in_2, in_3, "sum_first")


def replacement_func():
    """Returns the module-level replacement function."""
    return _dispatch_wrapper