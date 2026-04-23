import torch
import triton
import triton.language as tl

@triton.jit
def batch_norm_kernel(
    x_ptr, w_ptr, b_ptr, rm_ptr, rv_ptr,
    out_ptr,
    n_elements,
    channels,
    height,
    width,
    eps: tl.constexpr
):
    start_idx = tl.program_id(0) * 256
    offsets = start_idx + tl.arange(0, 256)
    mask = offsets < n_elements
    channel_idx = (offsets // (height * width)) % channels
    mean = tl.load(rm_ptr + channel_idx)
    var = tl.load(rv_ptr + channel_idx)
    weight = tl.load(w_ptr + channel_idx)
    bias = tl.load(b_ptr + channel_idx)
    x_val = tl.load(x_ptr + offsets, mask=mask)
    normalized = (x_val - mean) * weight / tl.sqrt(var + eps) + bias
    tl.store(out_ptr + offsets, normalized, mask=mask)

@torch.fx.wrap
def custom_batch_norm(x, running_mean, running_var, weight, bias):
    batch, channels, height, width = x.shape
    n_elements = batch * channels * height * width
    num_blocks = (n_elements + 255) // 256
    out = torch.empty_like(x)
    batch_norm_kernel[(num_blocks,)](
        x,
        weight,
        bias,
        running_mean,
        running_var,
        out,
        n_elements,
        channels,
        height,
        width,
        1e-05
    )
    return out

def pattern(x, running_mean, running_var, weight, bias):
    return torch.nn.functional.batch_norm(x, running_mean, running_var, weight, bias, False, 0.1, 1e-05)

def replacement_args(x, running_mean, running_var, weight, bias):
    return (x, running_mean, running_var, weight, bias)

def replacement_func():
    return custom_batch_norm