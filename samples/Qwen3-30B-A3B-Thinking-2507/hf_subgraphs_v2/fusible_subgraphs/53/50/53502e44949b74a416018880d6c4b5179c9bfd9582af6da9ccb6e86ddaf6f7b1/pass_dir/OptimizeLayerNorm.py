import torch
import triton
import triton.language as tl

# Pattern matching function

def pattern(x, weight, bias):
    # Match exactly the layer_norm call in the model
    return torch.nn.functional.layer_norm(x, (256,), weight, bias, 1e-05)

# Argument extraction function

def replacement_args(x, weight, bias):
    return (x, weight, bias)

# Triton kernel for layer norm
@triton.jit
def layer_norm_kernel(
    x_ptr,
    weight_ptr,
    bias_ptr,
    out_ptr,
    n_rows,
    n_cols,
    eps: tl.constexpr,
    BLOCK_SIZE: tl.constexpr
):
    # Each block processes one row
    row_idx = tl.program_id(0)
    row_start = row_idx * n_cols

    # Load entire row (256 elements)
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_cols
    x = tl.load(x_ptr + row_start + offsets, mask=mask, other=0.0)
    weight = tl.load(weight_ptr + offsets, mask=mask, other=0.0)
    bias = tl.load(bias_ptr + offsets, mask=mask, other=0.0)

    # Compute mean over row
    mean = tl.sum(x, axis=0) / n_cols

    # Compute variance
    x_centered = x - mean
    var = tl.sum(x_centered * x_centered, axis=0) / n_cols
    inv_std = 1.0 / tl.sqrt(var + eps)

    # Apply normalization
    x_norm = x_centered * inv_std
    out = x_norm * weight + bias

    # Store result
    tl.store(out_ptr + row_start + offsets, out, mask=mask)

# Kernel wrapper
@torch.fx.wrap
def layer_norm_wrapper(x, weight, bias):
    n_rows = x.numel() // x.shape[-1]  # Batch * Seq length
    n_cols = x.shape[-1]
    out = torch.empty_like(x)
    BLOCK_SIZE = 256  # Matches our fixed dimension
    grid = (n_rows,)

    layer_norm_kernel[grid](
        x_ptr=x,
        weight_ptr=weight,
        bias_ptr=bias,
        out_ptr=out,
        n_rows=n_rows,
        n_cols=n_cols,
        eps=1e-05,
        BLOCK_SIZE=BLOCK_SIZE
    )
    return out

# Replacement function

def replacement_func():
    return layer_norm_wrapper