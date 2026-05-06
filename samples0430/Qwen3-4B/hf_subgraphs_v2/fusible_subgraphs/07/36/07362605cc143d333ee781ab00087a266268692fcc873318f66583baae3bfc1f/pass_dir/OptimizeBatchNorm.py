import torch
import triton
import triton.language as tl

def pattern(input, running_mean, running_var, weight, bias, training=False, momentum=0.1, eps=1e-05):
    return torch.nn.functional.batch_norm(input, running_mean, running_var, weight, bias, training, momentum, eps)

def replacement_args(input, running_mean, running_var, weight, bias, training=False, momentum=0.1, eps=1e-05):
    return (input, running_mean, running_var, weight, bias)

@triton.jit
def batch_norm_kernel(
    input_ptr,
    running_mean_ptr,
    running_var_ptr,
    weight_ptr,
    bias_ptr,
    output_ptr,
    N,
    C,
    H,
    W,
    BLOCK_SIZE: tl.constexpr
):
    for i in tl.arange(0, C):
        for j in tl.arange(0, H):
            for k in tl.arange(0, W):
                idx = j * W + k
                input_val = tl.load(input_ptr + (i * H * W + idx))
                mean = tl.load(running_mean_ptr + i)
                var = tl.load(running_var_ptr + i)
                eps_val = 1e-6
                normalized = (input_val - mean) / tl.sqrt(var + eps_val)
                normalized = normalized * tl.load(weight_ptr + i) + tl.load(bias_ptr + i)
                tl.store(output_ptr + (i * H * W + idx), normalized)

def batch_norm_kernel_wrapper(input, running_mean, running_var, weight, bias):
    N = 1
    C = input.shape[1]
    H = input.shape[2]
    W = input.shape[3]
    output = torch.empty_like(input)
    batch_norm_kernel[(1,)](
        input_ptr=input,
        running_mean_ptr=running_mean,
        running_var_ptr=running_var,
        weight_ptr=weight,
        bias_ptr=bias,
        output_ptr=output,
        N=N,
        C=C,
        H=H,
        W=W,
        BLOCK_SIZE=16
    )
    return output

def replacement_func():
    return batch_norm_kernel_wrapper