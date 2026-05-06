import torch
import triton
import triton.language as tl

def pattern(input, weight, bias, running_mean, running_var, training, momentum, eps):
    return torch.nn.functional.batch_norm(input, weight, bias, running_mean, running_var, training, momentum, eps)

def replacement_args(input, weight, bias, running_mean, running_var, training, momentum, eps):
    return (input, weight, bias, running_mean, running_var, training, momentum, eps)

@triton.jit
def batch_norm_triton_kernel(
    input_ptr,
    weight_ptr,
    bias_ptr,
    running_mean_ptr,
    running_var_ptr,
    training_ptr,
    momentum_ptr,
    eps_ptr,
    N: tl.int32,
    C: tl.int32,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offset = pid * BLOCK_SIZE
    mask = tl.arange(0, BLOCK_SIZE) < N
    
    x = tl.load(input_ptr + offset, mask=mask, other=0.0)
    mean = tl.load(running_mean_ptr + offset, mask=mask, other=0.0)
    var = tl.load(running_var_ptr + offset, mask=mask, other=0.0)
    w = tl.load(weight_ptr + offset, mask=mask, other=1.0)
    b = tl.load(bias_ptr + offset, mask=mask, other=0.0)
    
    x = (x - mean) / tl.sqrt(var + eps_ptr) * w + b
    tl.store(input_ptr + offset, x, mask=mask)

@torch.fx.wrap
def batch_norm_triton_wrapper(
    input,
    weight,
    bias,
    running_mean,
    running_var,
    training=False,
    momentum=0.1,
    eps=1e-05,
):
    N, C = input.shape[:2]
    grid = (tl.cdiv(N * C, 1024),)
    output = torch.empty_like(input)
    batch_norm_triton_kernel[grid](
        input_ptr=input,
        weight_ptr=weight,
        bias_ptr=bias,
        running_mean_ptr=running_mean,
        running_var_ptr=running_var,
        training_ptr=training,
        momentum_ptr=momentum,
        eps_ptr=eps,
        N=N,
        C=C,
        BLOCK_SIZE=1024,
    )
    return output

def replacement_func():
    return batch_norm_triton_wrapper