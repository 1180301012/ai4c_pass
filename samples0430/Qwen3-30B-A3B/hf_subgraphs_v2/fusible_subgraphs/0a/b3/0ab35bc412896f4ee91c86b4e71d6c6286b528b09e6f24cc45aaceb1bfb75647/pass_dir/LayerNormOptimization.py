import torch
import triton
import triton.language as tl

# Pattern matching
def pattern(input, weight, bias):
    return torch.nn.functional.layer_norm(input, (128,), weight, bias, 1e-05)

def replacement_args(input, weight, bias):
    return (input, weight, bias)

@triton.jit
def layer_norm_kernel(
    input_ptr,
    weight_ptr,
    bias_ptr,
    output_ptr,
    hidden_size: tl.constexpr,
    eps: tl.float32,
):
    f = tl.program_id(0)
    # Unroll the sequence dimension (fixed at 4)
    idx0 = 0 * hidden_size + f
    x0 = tl.load(input_ptr + idx0)
    idx1 = 1 * hidden_size + f
    x1 = tl.load(input_ptr + idx1)
    idx2 = 2 * hidden_size + f
    x2 = tl.load(input_ptr + idx2)
    idx3 = 3 * hidden_size + f
    x3 = tl.load(input_ptr + idx3)
    sum_val = x0 + x1 + x2 + x3
    mean = sum_val / 4.0
    sum_sq = x0*x0 + x1*x1 + x2*x2 + x3*x3
    var = sum_sq / 4.0 - mean * mean
    inv_std = 1.0 / tl.sqrt(var + eps)
    w = tl.load(weight_ptr + f)
    b = tl.load(bias_ptr + f)
    y0 = (x0 - mean) * inv_std * w + b
    y1 = (x1 - mean) * inv_std * w + b
    y2 = (x2 - mean) * inv_std * w + b
    y3 = (x3 - mean) * inv_std * w + b
    tl.store(output_ptr + idx0, y0)
    tl.store(output_ptr + idx1, y1)
    tl.store(output_ptr + idx2, y2)
    tl.store(output_ptr + idx3, y3)

@torch.fx.wrap
def layer_norm_wrapper(input, weight, bias):
    # Input shape: [1,4,128]
    batch_size = input.shape[0]
    seq_len = input.shape[1]
    hidden_size = input.shape[2]
    output = torch.empty_like(input)
    grid = (hidden_size,)
    layer_norm_kernel[grid](
        input_ptr=input,
        weight_ptr=weight,
        bias_ptr=bias,
        output_ptr=output,
        hidden_size=hidden_size,
        eps=1e-5
    )
    return output

def replacement_func():
    return layer_norm_wrapper