import torch
import triton
import triton.language as tl

def pattern(x, normalized_shape, weight, bias, eps):
    return torch.nn.functional.layer_norm(x, normalized_shape, weight, bias, eps)

def replacement_args(x, normalized_shape, weight, bias, eps):
    return (x, normalized_shape, weight, bias, eps)

@triton.jit
def layer_norm_kernel(x_ptr, weight_ptr, bias_ptr, out_ptr, T, C, eps, BLOCK_SIZE: tl.constexpr):
    channel_id = tl.program_id(0)
    if channel_id >= C:
        return

    mean = tl.zeros(tl.float16)
    var = tl.zeros(tl.float16)
    
    # Process all time steps for this channel
    for t in range(T):
        x_val = tl.load(x_ptr + t * C + channel_id, tl.float16)
        mean += x_val
        var += x_val * x_val
    
    mean = mean / T
    var = var / T - mean * mean
    std = tl.sqrt(var + eps)
    
    # Apply normalization
    out_val = (tl.load(x_ptr + t * C + channel_id) - mean) / std
    
    # Scale and shift
    out_val = out_val * tl.load(weight_ptr + channel_id) + tl.load(bias_ptr + channel_id)
    
    tl.store(out_ptr + channel_id, out_val)

@torch.fx.wrap
def layer_norm_wrapper(x, normalized_shape, weight, bias, eps):
    C = normalized_shape[0]
    T = x.shape[1]
    out = torch.empty_like(x)
    
    layer_norm_kernel[tl.cdiv(C, 1024)](
        x_ptr=x,
        weight_ptr=weight,
        bias_ptr=bias,
        out_ptr=out,
        T=T,
        C=C,
        eps=eps,
        BLOCK_SIZE=1024,
    )
    return out

def replacement_func():
    return layer_norm_wrapper