import torch
import triton
import triton.language as tl

def pattern(x, normalized_shape, weight, bias, eps):
    return torch.nn.functional.layer_norm(x, normalized_shape, weight, bias, eps)

def replacement_args(x, normalized_shape, weight, bias, eps):
    C = normalized_shape[0]
    return (x, weight, bias, C)

@triton.jit
def layer_norm_kernel(
    x_ptr,
    weight_ptr,
    bias_ptr,
    out_ptr,
    N: tl.constexpr,
    C: tl.constexpr,
    eps: tl.constexpr
):
    seq_idx = tl.program_id(0)
    c = tl.thread_id(0)
    if c >= C:
        return
    sum_val = 0.0
    sum_sq = 0.0
    for i in range(C):
        val = tl.load(x_ptr + seq_idx * C + i)
        sum_val += val
        sum_sq += val * val
    mean_val = sum_val / C
    var_val = (sum_sq / C - mean_val * mean_val) + eps
    val = tl.load(x_ptr + seq_idx * C + c)
    normalized = (val - mean_val) * tl.rsqrt(var_val)
    out_val = normalized * tl.load(weight_ptr + c) + tl.load(bias_ptr + c)
    tl.store(out_ptr + seq_idx * C + c, out_val)

@torch.fx.wrap
def triton_layer_norm(x, weight, bias, C):
    N = x.shape[1]
    x_reshaped = x.reshape(N, C)
    out_reshaped = torch.empty_like(x_reshaped)
    layer_norm_kernel[(N,)](
        x_ptr=x_reshaped,
        weight_ptr=weight,
        bias_ptr=bias,
        out_ptr=out_reshaped,
        N=N,
        C=C,
        eps=1e-05
    )
    return out_reshaped.reshape(1, N, C)

def replacement_func():
    return triton_layer_norm