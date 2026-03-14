import torch
import triton
import triton.language as tl


def pattern(cat_input, normalized_shape, norm_weight, norm_bias):
    """
    Match layer_norm with any normalized_shape.
    """
    tmp_9 = torch.nn.functional.layer_norm(cat_input, normalized_shape, norm_weight, norm_bias, 1e-06)
    return tmp_9


def replacement_args(cat_input, normalized_shape, norm_weight, norm_bias):
    return (cat_input, norm_weight, norm_bias)


@triton.jit
def layernorm_kernel_128(
    input_ptr, weight_ptr, bias_ptr, output_ptr,
    n_rows, n_cols, input_row_stride, eps,
):
    BLOCK_SIZE: tl.constexpr = 128
    row_idx = tl.program_id(0)
    
    row_start = input_ptr + row_idx * input_row_stride
    out_start = output_ptr + row_idx * input_row_stride
    offs = tl.arange(0, BLOCK_SIZE)
    
    row = tl.load(row_start + offs).to(tl.float32)
    weight = tl.load(weight_ptr + offs).to(tl.float32)
    bias = tl.load(bias_ptr + offs).to(tl.float32)
    
    mean = tl.sum(row, axis=0) * (1.0 / 128.0)
    centered = row - mean
    var = tl.sum(centered * centered, axis=0) * (1.0 / 128.0)
    rstd = tl.rsqrt(var + eps)
    
    norm_row = centered * rstd * weight + bias
    tl.store(out_start + offs, norm_row)


@triton.jit
def layernorm_kernel_256(
    input_ptr, weight_ptr, bias_ptr, output_ptr,
    n_rows, n_cols, input_row_stride, eps,
):
    BLOCK_SIZE: tl.constexpr = 256
    row_idx = tl.program_id(0)
    
    row_start = input_ptr + row_idx * input_row_stride
    out_start = output_ptr + row_idx * input_row_stride
    offs = tl.arange(0, BLOCK_SIZE)
    
    row = tl.load(row_start + offs).to(tl.float32)
    weight = tl.load(weight_ptr + offs).to(tl.float32)
    bias = tl.load(bias_ptr + offs).to(tl.float32)
    
    mean = tl.sum(row, axis=0) * (1.0 / 256.0)
    centered = row - mean
    var = tl.sum(centered * centered, axis=0) * (1.0 / 256.0)
    rstd = tl.rsqrt(var + eps)
    
    norm_row = centered * rstd * weight + bias
    tl.store(out_start + offs, norm_row)


@triton.jit
def layernorm_kernel_512(
    input_ptr, weight_ptr, bias_ptr, output_ptr,
    n_rows, n_cols, input_row_stride, eps,
):
    BLOCK_SIZE: tl.constexpr = 512
    row_idx = tl.program_id(0)
    
    row_start = input_ptr + row_idx * input_row_stride
    out_start = output_ptr + row_idx * input_row_stride
    offs = tl.arange(0, BLOCK_SIZE)
    
    row = tl.load(row_start + offs).to(tl.float32)
    weight = tl.load(weight_ptr + offs).to(tl.float32)
    bias = tl.load(bias_ptr + offs).to(tl.float32)
    
    mean = tl.sum(row, axis=0) * (1.0 / 512.0)
    centered = row - mean
    var = tl.sum(centered * centered, axis=0) * (1.0 / 512.0)
    rstd = tl.rsqrt(var + eps)
    
    norm_row = centered * rstd * weight + bias
    tl.store(out_start + offs, norm_row)


@torch.fx.wrap
def optimized_layernorm(cat_input, norm_weight, norm_bias):
    """
    Optimized layer normalization.
    """
    shape = cat_input.shape
    n_rows = 1
    for s in shape[:-1]:
        n_rows *= s
    n_cols = shape[-1]
    
    cat_input = cat_input.contiguous()
    norm_weight = norm_weight.contiguous()
    norm_bias = norm_bias.contiguous()
    
    output = torch.empty_like(cat_input)
    
    input_flat = cat_input.view(-1, n_cols)
    output_flat = output.view(-1, n_cols)
    
    if n_cols == 128:
        layernorm_kernel_128[(n_rows,)](
            input_flat, norm_weight, norm_bias, output_flat,
            n_rows, n_cols, n_cols, 1e-6,
            num_warps=2,
        )
    elif n_cols == 256:
        layernorm_kernel_256[(n_rows,)](
            input_flat, norm_weight, norm_bias, output_flat,
            n_rows, n_cols, n_cols, 1e-6,
            num_warps=2,
        )
    else:
        layernorm_kernel_512[(n_rows,)](
            input_flat, norm_weight, norm_bias, output_flat,
            n_rows, n_cols, n_cols, 1e-6,
            num_warps=4,
        )
    
    return output


def replacement_func():
    return optimized_layernorm