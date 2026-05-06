import torch
import triton
import triton.language as tl

def pattern(x, normalized_shape, weight, bias, eps=1e-05):
    return torch.nn.functional.layer_norm(x, normalized_shape, weight, bias, eps)

def replacement_args(x, normalized_shape, weight, bias, eps=1e-05):
    return (x, weight, bias, eps)

def replacement_func():
    return layer_norm_kernel_wrapper

@triton.jit
def layer_norm_kernel(x_ptr, weight_ptr, bias_ptr, N, S, C, EPS: float):
    c = tl.program_id(0)
    if c >= C:
        return
    
    # Load 4 values for this feature (N=1, S=4)
    x_vals = [tl.load(x_ptr + i * C + c) for i in range(N * S)]
    mean = tl.sum(x_vals) / (N * S)
    variance = tl.sum(tl.square(x_vals)) / (N * S) - mean * mean
    std = tl.sqrt(variance + EPS)
    
    # Normalize and apply weight/bias
    normalized_vals = (tl.tensor(x_vals) - mean) / std
    output_vals = normalized_vals * tl.load(weight_ptr + c) + tl.load(bias_ptr + c)
    
    # Store results back
    for i in range(N * S):
        tl.store(x_ptr + i * C + c, output_vals[i])

@torch.fx.wrap
def layer_norm_kernel_wrapper(x, weight, bias, eps=1e-05):
    B, S, C = x.shape
    out = torch.empty_like(x)
    layer_norm_kernel[(C,)](\n        x_ptr=x,
        weight_ptr=weight,
        bias_ptr=bias,
        N=B,
        S=S,
        C=C,
        EPS=eps,
    )
    return out