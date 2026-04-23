import torch
import triton
import triton.language as tl


@triton.jit
def layer_norm_kernel(
    x_ptr, weight_ptr, bias_ptr, out_ptr,
    N: tl.constexpr, seq_len: tl.constexpr, hidden: tl.constexpr,
    eps: tl.constexpr, BLOCK: tl.constexpr
):
    """
    Optimized layer norm kernel for hidden=768.
    """
    row_pid = tl.program_id(0)
    
    # Compute mean and var for this row
    base_offset = row_pid * hidden
    
    sum_val = 0.0
    sum_sq = 0.0
    
    # Sequential reduce over hidden dim
    for i in range(hidden):
        offset = base_offset + i
        val = tl.load(x_ptr + offset, mask=row_pid * hidden + i < N * seq_len * hidden, other=0.0)
        sum_val += val
        sum_sq += val * val
    
    mean = sum_val / hidden
    var = (sum_sq / hidden) - (mean * mean)
    inv_std = tl.rsqrt(var + eps)
    
    # Compute output
    for i in range(hidden):
        offset = base_offset + i
        mask_val = row_pid * hidden + i < N * seq_len * hidden
        val = tl.load(x_ptr + offset, mask=mask_val, other=0.0)
        ln_val = (val - mean) * inv_std
        w = tl.load(weight_ptr + i, mask=i < hidden, other=1.0)
        b = tl.load(bias_ptr + i, mask=i < hidden, other=0.0)
        out = ln_val * w + b
        tl.store(out_ptr + offset, out, mask=mask_val)


@torch.fx.wrap
def fused_layer_norm(x, weight, bias, normalized_shape, eps):
    """
    Fused layer norm kernel.
    """
    N = x.shape[0]
    seq_len = x.shape[1]
    hidden = x.shape[2]
    
    out = torch.empty_like(x)
    
    num_rows = N * seq_len
    BLOCK_SIZE = 256
    
    layer_norm_kernel[(num_rows,)](
        x, weight, bias, out,
        N, seq_len, hidden, eps, BLOCK_SIZE
    )
    
    return out


def pattern(x, weight, bias):
    """
    Match torch.nn.functional.layer_norm with normalized_shape=(768,)
    Returns the layer norm result.
    """
    result = torch.nn.functional.layer_norm(x, (768,), weight, bias, 1e-06)
    return result


def replacement_args(x, weight, bias):
    return (x, weight, bias, 768, 1e-06)


def replacement_func():
    return fused_layer_norm