import torch
import triton
import triton.language as tl
import operator

# Pattern for graph 3: [1, 5, 7, 5, 7, 384] -> 35x35x384 -> roll -> 32x32 -> 1024x384
def pattern(in_0, in_1, in_2, in_3):
    tmp_0 = in_0
    tmp_1 = in_1
    tmp_2 = in_3.contiguous()
    tmp_3 = tmp_2.view(-1, 35, 35, 384)
    tmp_4 = torch.roll(tmp_3, shifts=(3, 3), dims=(1, 2))
    tmp_5 = tmp_4[slice(None, None, None), slice(None, 32, None), slice(None, 32, None), slice(None, None, None)]
    tmp_6 = tmp_5.contiguous()
    tmp_7 = tmp_6.view(1, 1024, 384)
    tmp_8 = in_2 + tmp_7
    tmp_9 = torch.nn.functional.layer_norm(tmp_8, (384,), tmp_1, tmp_0, 1e-05)
    return (tmp_8, tmp_9)

def replacement_args(in_0, in_1, in_2, in_3):
    return (in_0, in_1, in_2, in_3)

@triton.jit
def fused_kernel_384(
    x_ptr, residual_ptr, weight_ptr, bias_ptr,
    out_add_ptr, out_ln_ptr,
    H_in: tl.constexpr, W_in: tl.constexpr,
    H_out: tl.constexpr, W_out: tl.constexpr,
    C: tl.constexpr, shift: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    row = tl.program_id(0)
    h = row // W_out
    w = row % W_out
    src_h = (h - shift + H_in) % H_in
    src_w = (w - shift + W_in) % W_in
    x_offset = src_h * W_in * C + src_w * C
    
    col_offsets = tl.arange(0, BLOCK_C)
    mask = col_offsets < C
    x_vals = tl.load(x_ptr + x_offset + col_offsets, mask=mask, other=0.0)
    
    res_offset = row * C
    res_vals = tl.load(residual_ptr + res_offset + col_offsets, mask=mask, other=0.0)
    
    add_result = res_vals + x_vals
    tl.store(out_add_ptr + res_offset + col_offsets, add_result, mask=mask)
    
    # Layer normalization - ensure masked elements don't affect computation
    add_result_fp32 = add_result.to(tl.float32)
    add_result_masked = tl.where(mask, add_result_fp32, 0.0)
    mean = tl.sum(add_result_masked, axis=0) / C
    centered = tl.where(mask, add_result_fp32 - mean, 0.0)
    var = tl.sum(centered * centered, axis=0) / C
    inv_std = 1.0 / tl.sqrt(var + 1e-5)
    normalized = centered * inv_std
    
    weight = tl.load(weight_ptr + col_offsets, mask=mask, other=1.0)
    bias = tl.load(bias_ptr + col_offsets, mask=mask, other=0.0)
    ln_result = normalized * weight + bias
    tl.store(out_ln_ptr + res_offset + col_offsets, ln_result.to(add_result.dtype), mask=mask)

@torch.fx.wrap
def run_fused_kernel(bias, weight, residual, x):
    H_in, W_in = 35, 35
    H_out, W_out = 32, 32
    C = 384
    x_contig = x.contiguous().view(-1)
    N_out = H_out * W_out
    out_add = torch.empty(1, N_out, C, device=x.device, dtype=x.dtype)
    out_ln = torch.empty(1, N_out, C, device=x.device, dtype=x.dtype)
    
    grid = (N_out,)
    fused_kernel_384[grid](
        x_contig, residual, weight, bias,
        out_add, out_ln,
        H_in, W_in, H_out, W_out, C, 3, 512
    )
    return (out_add, out_ln)

def fused_replacement(in_0, in_1, in_2, in_3):
    result = run_fused_kernel(in_0, in_1, in_2, in_3)
    out_add = operator.getitem(result, 0)
    out_ln = operator.getitem(result, 1)
    return (out_add, out_ln)

def replacement_func():
    return fused_replacement