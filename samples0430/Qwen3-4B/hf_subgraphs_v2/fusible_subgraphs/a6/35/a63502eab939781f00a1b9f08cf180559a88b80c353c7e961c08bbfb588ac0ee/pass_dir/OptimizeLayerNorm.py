import torch
import triton
import triton.language as tl

def pattern(input, weight, bias, eps=1e-05):
    return torch.nn.functional.layer_norm(input, (768,), weight, bias, eps)

def replacement_args(input, weight, bias, eps=1e-05):
    return (input, weight, bias, eps)

@triton.jit
def layer_norm_kernel(
    input_ptr,
    weight_ptr,
    bias_ptr,
    eps: tl.float16,
    n_channels: tl.int32,
    n_features: tl.int32,
):
    channel_id = tl.program_id(0)
    offsets = tl.arange(0, n_features)
    input_data = tl.load(input_ptr + channel_id * n_features + offsets)
    mean = tl.sum(input_data) / n_features
    var = tl.sum((input_data - mean) ** 2) / n_features
    normalized = (input_data - mean) / tl.sqrt(var + eps)
    output_data = normalized * tl.load(weight_ptr + channel_id) + tl.load(bias_ptr + channel_id)
    tl.store(input_ptr + channel_id * n_features + offsets, output_data)

@torch.fx.wrap
def kernel_wrapper(input, weight, bias, eps=1e-05):
    n_channels = weight.size(0)
    n_features = input.size(-1)
    output = torch.empty_like(input)
    layer_norm_kernel[(n_channels,)](
    input_ptr=input.data_ptr(),
    weight_ptr=weight.data_ptr(),
    bias_ptr=bias.data_ptr(),
    eps=eps,
    n_channels=n_channels,
    n_features=n_features,
)
    return output

def replacement_func():
    return kernel_wrapper