import torch
import triton
import triton.language as tl

def pattern(x, weight, bias):
    return torch.nn.functional.layer_norm(x, (1024,), weight, bias, 1e-05)

def replacement_args(x, weight, bias):
    return (x, weight, bias)

@triton.jit
def layer_norm_kernel(
    x_ptr,
    weight_ptr,
    bias_ptr,
    out_ptr,
    n_elements: tl.int32,
    channels: tl.int32,
    eps: tl.float32 = 1e-05,
    BLOCK_SIZE: tl.constexpr = 1024
):
    """
    Triton kernel for layer normalization.
    Input: x [batch, channels, ...]
    """
    # Calculate the index in the input and respective output
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    # Load data from input
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    # Compute mean over channels
    mean = tl.reduce(x, axis=0, init=0.0)
    # Compute variance
    var = tl.sum((x - mean) ** 2, axis=0)
    # Normalize
    normalized = (x - mean) / tl.sqrt(var + eps)
    # Apply weight and bias
    out = normalized * tl.load(weight_ptr + offsets) + tl.load(bias_ptr + offsets)
    # Store output
    tl.store(out_ptr + offsets, out, mask=mask)

def layer_norm_wrapper(x, weight, bias):
    n_elements = x.numel()
    channels = x.shape[1]
    out = torch.empty_like(x)
    layer_norm_kernel[(1,)](
        x_ptr=x,
        weight_ptr=weight,
        bias_ptr=bias,
        out_ptr=out,
        n_elements=n_elements,
        channels=channels,
    )
    return out



def replacement_func():
    return layer_norm_wrapper