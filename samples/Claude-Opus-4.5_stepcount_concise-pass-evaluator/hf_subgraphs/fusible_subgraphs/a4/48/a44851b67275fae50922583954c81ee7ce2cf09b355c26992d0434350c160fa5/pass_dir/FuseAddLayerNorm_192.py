import torch
import triton
import triton.language as tl

# Pattern that matches add + layer_norm with any normalized_shape
def pattern(in_0, in_1, in_2, in_3, normalized_shape):
    tmp_2 = in_2 + in_3
    tmp_3 = torch.nn.functional.layer_norm(tmp_2, normalized_shape, in_1, in_0, 1e-06)
    return tmp_2, tmp_3

def replacement_args(in_0, in_1, in_2, in_3, normalized_shape):
    return (in_0, in_1, in_2, in_3)

# Kernel optimized for smaller N (<=256)
@triton.jit
def fused_add_layernorm_kernel_small(
    in_2_ptr, in_3_ptr, weight_ptr, bias_ptr,
    out_sum_ptr, out_norm_ptr,
    M, N, eps,
):
    row_idx = tl.program_id(0)
    row_start = row_idx * N
    cols = tl.arange(0, 256)
    mask = cols < N
    
    in_2 = tl.load(in_2_ptr + row_start + cols, mask=mask, other=0.0).to(tl.float32)
    in_3 = tl.load(in_3_ptr + row_start + cols, mask=mask, other=0.0).to(tl.float32)
    x = in_2 + in_3
    
    tl.store(out_sum_ptr + row_start + cols, x, mask=mask)
    
    x_sum = tl.sum(tl.where(mask, x, 0.0), axis=0)
    mean = x_sum / N
    
    x_centered = x - mean
    var = tl.sum(tl.where(mask, x_centered * x_centered, 0.0), axis=0) / N
    rstd = tl.rsqrt(var + eps)
    
    weight = tl.load(weight_ptr + cols, mask=mask, other=1.0).to(tl.float32)
    bias = tl.load(bias_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    
    out = x_centered * rstd * weight + bias
    tl.store(out_norm_ptr + row_start + cols, out, mask=mask)

# Kernel optimized for larger N (256<N<=512)
@triton.jit
def fused_add_layernorm_kernel_large(
    in_2_ptr, in_3_ptr, weight_ptr, bias_ptr,
    out_sum_ptr, out_norm_ptr,
    M, N, eps,
):
    row_idx = tl.program_id(0)
    row_start = row_idx * N
    cols = tl.arange(0, 512)
    mask = cols < N
    
    in_2 = tl.load(in_2_ptr + row_start + cols, mask=mask, other=0.0).to(tl.float32)
    in_3 = tl.load(in_3_ptr + row_start + cols, mask=mask, other=0.0).to(tl.float32)
    x = in_2 + in_3
    
    tl.store(out_sum_ptr + row_start + cols, x, mask=mask)
    
    x_sum = tl.sum(tl.where(mask, x, 0.0), axis=0)
    mean = x_sum / N
    
    x_centered = x - mean
    var = tl.sum(tl.where(mask, x_centered * x_centered, 0.0), axis=0) / N
    rstd = tl.rsqrt(var + eps)
    
    weight = tl.load(weight_ptr + cols, mask=mask, other=1.0).to(tl.float32)
    bias = tl.load(bias_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    
    out = x_centered * rstd * weight + bias
    tl.store(out_norm_ptr + row_start + cols, out, mask=mask)

@torch.fx.wrap
def _fused_add_layernorm_impl(in_0, in_1, in_2, in_3):
    shape = in_2.shape
    N = shape[-1]
    M = in_2.numel() // N
    
    out_sum = torch.empty_like(in_2)
    out_norm = torch.empty_like(in_2)
    
    device = in_2.device
    weight = in_1.to(device) if in_1.device != device else in_1
    bias = in_0.to(device) if in_0.device != device else in_0
    
    if N <= 256:
        fused_add_layernorm_kernel_small[(M,)](
            in_2, in_3, weight, bias,
            out_sum, out_norm,
            M, N, 1e-06,
            num_warps=4,
        )
    else:
        fused_add_layernorm_kernel_large[(M,)](
            in_2, in_3, weight, bias,
            out_sum, out_norm,
            M, N, 1e-06,
            num_warps=8,
        )
    
    return out_sum, out_norm

def fused_add_layernorm(in_0, in_1, in_2, in_3):
    result = _fused_add_layernorm_impl(in_0, in_1, in_2, in_3)
    out_sum = result[0]
    out_norm = result[1]
    return out_sum, out_norm

def replacement_func():
    return fused_add_layernorm