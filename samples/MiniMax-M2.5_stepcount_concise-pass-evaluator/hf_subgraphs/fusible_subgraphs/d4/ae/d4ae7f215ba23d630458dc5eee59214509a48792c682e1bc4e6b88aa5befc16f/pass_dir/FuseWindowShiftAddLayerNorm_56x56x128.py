import torch
import triton
import triton.language as tl


@triton.jit
def triton_layernorm_kernel(input_ptr, weight_ptr, bias_ptr, output_ptr,
                            N: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    """Triton layer norm kernel"""
    pid = tl.program_id(0)
    offs = tl.arange(0, BLOCK_SIZE)
    mask = offs < N
    
    x = tl.load(input_ptr + pid * N + offs, mask=mask, other=0.0)
    mean = tl.sum(x, axis=0) / N
    diff = x - mean
    variance = tl.sum(diff * diff, axis=0) / N
    std = tl.sqrt(variance + 1e-5)
    normalized = diff / std
    
    w = tl.load(weight_ptr + offs, mask=mask, other=0.0)
    b = tl.load(bias_ptr + offs, mask=mask, other=0.0)
    result = normalized * w + b
    
    tl.store(output_ptr + pid * N + offs, result, mask=mask)


def pattern(x, weight, bias):
    """Match layer_norm only"""
    return torch.nn.functional.layer_norm(x, (128,), weight, bias, 1e-05)


def replacement_args(x, weight, bias):
    return (x, weight, bias)


@torch.fx.wrap
def triton_layernorm_wrapper(x, weight, bias):
    """Triton layer norm - replaces PyTorch's layer_norm"""
    N = 128
    SEQ_LEN = x.shape[1]
    output = torch.empty_like(x)
    
    grid = (SEQ_LEN,)
    triton_layernorm_kernel[grid](x, weight, bias, output, N, N)
    
    # Return just the output - the graph will handle passing through x
    return output


def replacement_func():
    return triton_layernorm_wrapper