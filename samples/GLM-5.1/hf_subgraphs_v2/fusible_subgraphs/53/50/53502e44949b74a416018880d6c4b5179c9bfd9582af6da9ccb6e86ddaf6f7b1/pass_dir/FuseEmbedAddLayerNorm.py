import torch
import triton
import triton.language as tl

from torch import device

# Simple pattern: just the layer_norm part
def pattern(in_10, in_3, in_2):
    tmp_11 = torch.nn.functional.layer_norm(in_10, (256,), in_3, in_2, 1e-05)
    return tmp_11

def replacement_args(in_10, in_3, in_2):
    return (in_10, in_3, in_2, "layernorm_256")

@triton.jit
def layernorm_kernel(
    input_ptr,
    weight_ptr,
    bias_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    x = tl.load(input_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    w = tl.load(weight_ptr + offsets, mask=mask, other=1.0).to(tl.float32)
    b = tl.load(bias_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    
    mean = tl.sum(x, axis=0) / n_elements
    centered = x - mean
    variance = tl.sum(centered * centered, axis=0) / n_elements
    rstd = 1.0 / tl.sqrt(variance + 1e-05)
    normalized = centered * rstd
    output = normalized * w + b
    
    tl.store(out_ptr + offsets, output, mask=mask)


@torch.fx.wrap
def triton_layernorm_256(input, weight, bias):
    n_elements = input.numel()
    BLOCK_SIZE = 256
    out = torch.empty_like(input)
    
    layernorm_kernel[(1,)](
        input_ptr=input,
        weight_ptr=weight,
        bias_ptr=bias,
        out_ptr=out,
        n_elements=n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return out


@torch.fx.wrap
def dispatch_wrapper(*args):
    route = args[-1]
    if route == "layernorm_256":
        return triton_layernorm_256(args[0], args[1], args[2])
    else:
        raise RuntimeError(f"Unknown route: {route}")

def replacement_func():
    return dispatch_wrapper