import torch
import triton
import triton.language as tl

def pattern(in_2, in_3, in_1, in_0):
    tmp_2 = in_2 + in_3
    tmp_4 = torch.nn.functional.layer_norm(tmp_2, (1024,), in_1, in_0, 1e-05)
    return (tmp_2, tmp_4)

def replacement_args(in_2, in_3, in_1, in_0):
    return (in_2, in_3, in_1, in_0)

@triton.jit
def fused_add_layer_norm_kernel(
    in_2_ptr, in_3_ptr, weight_ptr, bias_ptr,
    sum_out_ptr, norm_out_ptr,
    S: tl.constexpr,
    C: tl.constexpr = 1024,
    eps: tl.constexpr = 1e-05,
    BLOCK_SIZE: tl.constexpr = 1024
):
    s = tl.program_id(0)
    start_c = 0
    mask = tl.arange(0, BLOCK_SIZE) < C
    
    in_2 = tl.load(in_2_ptr + s * C, mask=mask, other=0.0)
    in_3 = tl.load(in_3_ptr + s * C, mask=mask, other=0.0)
    sum_val = in_2 + in_3
    tl.store(sum_out_ptr + s * C, sum_val, mask=mask)

    sum1 = tl.sum(sum_val, axis=0)
    sum2 = tl.sum(sum_val * sum_val, axis=0)
    mean = sum1 / C
    var = sum2 / C - mean * mean
    var = var + eps
    std = tl.sqrt(var)
    normalized = (sum_val - mean) / std
    weight = tl.load(weight_ptr, mask=mask, other=0.0)
    bias = tl.load(bias_ptr, mask=mask, other=0.0)
    norm_out = normalized * weight + bias
    tl.store(norm_out_ptr + s * C, norm_out, mask=mask)

@torch.fx.wrap
def fused_add_layer_norm(in_2, in_3, in_1, in_0):
    S = in_2.shape[1]
    C = in_2.shape[2]
    sum_out = torch.empty_like(in_2)
    norm_out = torch.empty_like(in_2)
    grid = (S,)
    fused_add_layer_norm_kernel[grid](
        in_2, in_3, in_1, in_0, sum_out, norm_out, S, C, 1e-05, C
    )
    return sum_out, norm_out

def replacement_func():
    return fused_add_layer_norm