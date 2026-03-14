import torch
import triton
import triton.language as tl


def pattern(cat_input, norm_weight, norm_bias):
    """
    Match the computation pattern:
    - layer_norm((512,))
    """
    tmp_9 = torch.nn.functional.layer_norm(cat_input, (512,), norm_weight, norm_bias, 1e-06)
    return tmp_9


def replacement_args(cat_input, norm_weight, norm_bias):
    return (cat_input, norm_weight, norm_bias)


@triton.jit
def layernorm_kernel_512(
    input_ptr,
    weight_ptr,
    bias_ptr,
    output_ptr,
    n_rows,
    n_cols,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Optimized layer norm kernel.
    Each program handles one row.
    """
    row_idx = tl.program_id(0)
    
    # Pointers to the row
    row_start = input_ptr + row_idx * n_cols
    out_start = output_ptr + row_idx * n_cols
    
    offs = tl.arange(0, BLOCK_SIZE)
    mask = offs < n_cols
    
    # Load row
    row = tl.load(row_start + offs, mask=mask, other=0.0)
    
    # Compute mean
    row_sum = tl.sum(row, axis=0)
    mean = row_sum / n_cols
    
    # Compute variance
    centered = row - mean
    var_sum = tl.sum(centered * centered, axis=0)
    var = var_sum / n_cols
    rstd = 1.0 / tl.sqrt(var + eps)
    
    # Load weight and bias
    weight = tl.load(weight_ptr + offs, mask=mask, other=1.0)
    bias = tl.load(bias_ptr + offs, mask=mask, other=0.0)
    
    # Apply normalization
    norm_row = centered * rstd * weight + bias
    
    # Store
    tl.store(out_start + offs, norm_row, mask=mask)


@torch.fx.wrap
def optimized_layernorm_512(cat_input, norm_weight, norm_bias):
    """
    Optimized layer normalization.
    """
    # Get dimensions
    shape = cat_input.shape
    n_rows = 1
    for s in shape[:-1]:
        n_rows *= s
    n_cols = shape[-1]
    
    # Ensure contiguous
    cat_input = cat_input.contiguous()
    norm_weight = norm_weight.contiguous()
    norm_bias = norm_bias.contiguous()
    
    # Output tensor
    output = torch.empty_like(cat_input)
    
    # Launch kernel
    BLOCK_SIZE = 512
    
    layernorm_kernel_512[(n_rows,)](
        cat_input,
        norm_weight,
        norm_bias,
        output,
        n_rows,
        n_cols,
        1e-6,
        BLOCK_SIZE,
    )
    
    return output


def replacement_func():
    return optimized_layernorm_512