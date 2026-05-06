import torch
import triton
import triton.language as tl

def pattern(input, running_mean, running_var, weight, bias):
    return torch.nn.functional.batch_norm(
        input, running_mean, running_var, weight, bias,
        training=False, momentum=0.1, eps=1e-05
    )

def replacement_args(input, running_mean, running_var, weight, bias):
    return (input, running_mean, running_var, weight, bias)

@triton.jit
def batchnorm_kernel(
    input_ptr,
    running_mean_ptr,
    running_var_ptr,
    weight_ptr,
    bias_ptr,
    output_ptr,
    H: tl.int32,
    W: tl.int32,
    C: tl.int32,
    eps: tl.float16,
    BLOCK_SIZE: tl.constexpr,
):
    c = tl.program_id(0)
    if c >= C:
        return
    # Compute mean for current channel across H*W
    mean_val = tl.zeros(tl.float16, tl.int32(1))
    for i in tl.arange(0, H * W):
        index = i
        x_val = tl.load(input_ptr + (c * H * W + index), mask=tl.arange(0, H*W) < (H*W), other=0.0)
        mean_val += x_val
    mean_val /= H * W
    
    # Compute variance for current channel
    var_val = tl.zeros(tl.float16, tl.int32(1))
    for i in tl.arange(0, H * W):
        index = i
        x_val = tl.load(input_ptr + (c * H * W + index), mask=tl.arange(0, H*W) < (H*W), other=0.0)
        x_diff = x_val - mean_val
        var_val += x_diff * x_diff
    var_val /= H * W
    
    # Normalize and apply scale/shift
    sqrt_var = tl.sqrt(var_val + eps)
    for i in tl.arange(0, H * W):
        index = i
        x_val = tl.load(input_ptr + (c * H * W + index), mask=tl.arange(0, H*W) < (H*W), other=0.0)
        x_norm = (x_val - mean_val) / sqrt_var
        weight_val = tl.load(weight_ptr + c, mask=tl.arange(0, C) < C, other=0.0)
        bias_val = tl.load(bias_ptr + c, mask=tl.arange(0, C) < C, other=0.0)
        out_val = weight_val * x_norm + bias_val
        tl.store(output_ptr + (c * H * W + index), out_val)

@torch.fx.wrap
def batchnorm_wrapper(input, running_mean, running_var, weight, bias):
    N, C, H, W = input.shape
    output = torch.empty_like(input)
    batchnorm_kernel[(C,)](
        input_ptr=input,
        running_mean_ptr=running_mean,
        running_var_ptr=running_var,
        weight_ptr=weight,
        bias_ptr=bias,
        output_ptr=output,
        H=H,
        W=W,
        C=C,
        eps=1e-05,
        BLOCK_SIZE=128
    )
    return output

def replacement_func():
    return batchnorm_wrapper