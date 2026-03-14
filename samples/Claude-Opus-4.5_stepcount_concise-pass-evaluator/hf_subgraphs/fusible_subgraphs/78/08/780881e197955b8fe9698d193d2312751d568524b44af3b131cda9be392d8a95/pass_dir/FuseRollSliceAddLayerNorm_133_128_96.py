import torch
import triton
import triton.language as tl
import operator

# Pattern for graph 1: [1, 19, 7, 19, 7, 96] -> 133x133x96 -> roll -> 128x128 -> 16384x96
def pattern(in_0, in_1, in_2, in_3):
    tmp_0 = in_0
    tmp_1 = in_1
    tmp_2 = in_3.contiguous()
    tmp_3 = tmp_2.view(-1, 133, 133, 96)
    tmp_4 = torch.roll(tmp_3, shifts=(3, 3), dims=(1, 2))
    tmp_5 = tmp_4[slice(None, None, None), slice(None, 128, None), slice(None, 128, None), slice(None, None, None)]
    tmp_6 = tmp_5.contiguous()
    tmp_7 = tmp_6.view(1, 16384, 96)
    tmp_8 = in_2 + tmp_7
    tmp_9 = torch.nn.functional.layer_norm(tmp_8, (96,), tmp_1, tmp_0, 1e-05)
    return (tmp_8, tmp_9)

def replacement_args(in_0, in_1, in_2, in_3):
    return (in_0, in_1, in_2, in_3)

@triton.autotune(
    configs=[
        triton.Config({}, num_warps=2),
        triton.Config({}, num_warps=4),
        triton.Config({}, num_warps=8),
    ],
    key=['C'],
)
@triton.jit
def fused_kernel_96(
    x_ptr, residual_ptr, weight_ptr, bias_ptr,
    out_add_ptr, out_ln_ptr,
    N_out, H_in: tl.constexpr, W_in: tl.constexpr,
    H_out: tl.constexpr, W_out: tl.constexpr,
    C: tl.constexpr, shift: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    row = tl.program_id(0)
    
    # Compute source position after roll
    h = row // W_out
    w = row % W_out
    src_h = (h - shift + H_in) % H_in
    src_w = (w - shift + W_in) % W_in
    x_offset = src_h * W_in * C + src_w * C
    
    col_offsets = tl.arange(0, BLOCK_C)
    mask = col_offsets < C
    
    # Load data
    x_vals = tl.load(x_ptr + x_offset + col_offsets, mask=mask, other=0.0)
    res_offset = row * C
    res_vals = tl.load(residual_ptr + res_offset + col_offsets, mask=mask, other=0.0)
    
    # Add
    add_result = res_vals + x_vals
    tl.store(out_add_ptr + res_offset + col_offsets, add_result, mask=mask)
    
    # Layer normalization in float32
    add_result_fp32 = add_result.to(tl.float32)
    
    # Compute mean - only over valid elements
    add_result_masked = tl.where(mask, add_result_fp32, 0.0)
    mean = tl.sum(add_result_masked, axis=0) * (1.0 / C)
    
    # Compute variance
    centered = tl.where(mask, add_result_fp32 - mean, 0.0)
    var = tl.sum(centered * centered, axis=0) * (1.0 / C)
    
    # Normalize using rsqrt for better performance
    inv_std = tl.rsqrt(var + 1e-5)
    normalized = centered * inv_std
    
    # Load weight and bias, apply scale and shift
    weight = tl.load(weight_ptr + col_offsets, mask=mask, other=1.0)
    bias = tl.load(bias_ptr + col_offsets, mask=mask, other=0.0)
    ln_result = normalized * weight + bias
    
    tl.store(out_ln_ptr + res_offset + col_offsets, ln_result.to(add_result.dtype), mask=mask)

@torch.fx.wrap
def run_fused_kernel(bias, weight, residual, x):
    H_in, W_in = 133, 133
    H_out, W_out = 128, 128
    C = 96
    BLOCK_C = 128
    
    x_contig = x.contiguous().view(-1)
    N_out = H_out * W_out
    out_add = torch.empty(1, N_out, C, device=x.device, dtype=x.dtype)
    out_ln = torch.empty(1, N_out, C, device=x.device, dtype=x.dtype)
    
    grid = (N_out,)
    fused_kernel_96[grid](
        x_contig, residual, weight, bias,
        out_add, out_ln,
        N_out, H_in, W_in, H_out, W_out, C, 3, BLOCK_C
    )
    return (out_add, out_ln)

def fused_replacement(in_0, in_1, in_2, in_3):
    result = run_fused_kernel(in_0, in_1, in_2, in_3)
    out_add = operator.getitem(result, 0)
    out_ln = operator.getitem(result, 1)
    return (out_add, out_ln)

def replacement_func():
    return fused_replacement