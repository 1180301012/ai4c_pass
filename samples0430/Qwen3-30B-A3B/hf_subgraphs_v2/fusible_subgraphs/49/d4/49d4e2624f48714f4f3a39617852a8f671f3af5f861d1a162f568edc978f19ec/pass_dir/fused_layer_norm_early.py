import torch
import triton
import triton.language as tl

@triton.jit
def fused_layer_norm_kernel(
    x_ptr,
    y_ptr,
    weight_ptr,
    bias_ptr,
    out_ptr,
    n_elements,
    eps: tl.constexpr,
    BLOCK_SIZE: tl.constexpr
):
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = tl.load(y_ptr + offsets, mask=mask, other=0.0)
    weight = tl.load(weight_ptr + offsets, mask=mask, other=0.0)
    bias = tl.load(bias_ptr + offsets, mask=mask, other=0.0)
    
    avg = (x + y) * 0.5
    
    sum_avg = tl.sum(avg, axis=0)
    sum_avg_sq = tl.sum(avg * avg, axis=0)
    
    mean_val = sum_avg / n_elements
    var_val = (sum_avg_sq / n_elements) - (mean_val * mean_val)
    std_val = tl.sqrt(var_val + eps)
    
    norm = (avg - mean_val) / (std_val + eps)
    output = norm * weight + bias
    
    tl.store(out_ptr + offsets, output, mask=mask)

@torch.fx.wrap
def fused_layer_norm(x, y, weight, bias, eps=1e-12):
    n_elements = x.numel()
    BLOCK_SIZE = 128
    num_blocks = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    grid = (num_blocks,)
    
    out = torch.empty_like(x)
    fused_layer_norm_kernel[grid](
        x_ptr=x,
        y_ptr=y,
        weight_ptr=weight,
        bias_ptr=bias,
        out_ptr=out,
        n_elements=n_elements,
        eps=eps,
        BLOCK_SIZE=BLOCK_SIZE
    )
    return out

def pattern(in_2, in_3, in_1, in_0):
    tmp_2 = in_2 + in_3
    tmp_3 = tmp_2 / 2
    tmp_4 = torch.nn.functional.layer_norm(tmp_3, (768,), in_1, in_0, 1e-12)
    return tmp_4

def replacement_args(in_2, in_3, in_1, in_0):
    return (in_2, in_3, in_1, in_0)

def replacement_func():
    return fused_layer_norm