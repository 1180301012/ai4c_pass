import torch
import triton
import triton.language as tl

def pattern(input, running_mean, running_var, weight, bias):
    return torch.nn.functional.batch_norm(input, running_mean, running_var, weight, bias, False, 0.1, 1e-05)

def replacement_args(input, running_mean, running_var, weight, bias):
    return (input, running_mean, running_var, weight, bias)

@triton.jit
def batch_norm_kernel(
    input_ptr,
    running_mean_ptr,
    running_var_ptr,
    weight_ptr,
    bias_ptr,
    eps: tl.float32,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < tl.minimum(BLOCK_SIZE, input_ptr.shape[0])
    input_vals = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    running_mean_val = tl.load(running_mean_ptr)
    running_var_val = tl.load(running_var_ptr)
    weight_val = tl.load(weight_ptr)
    bias_val = tl.load(bias_ptr)
    normalized_vals = (input_vals - running_mean_val) / tl.sqrt(running_var_val + eps) * weight_val + bias_val
    tl.store(input_ptr + offsets, normalized_vals, mask=mask)

def batch_norm_wrapper(input, running_mean, running_var, weight, bias):
    n_elements = input.numel()
    BLOCK_SIZE = 256
    num_programs = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    batch_norm_kernel[(num_programs,)](
        input_ptr=input,
        running_mean_ptr=running_mean,
        running_var_ptr=running_var,
        weight_ptr=weight,
        bias_ptr=bias,
        eps=1e-05,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return input

def replacement_func():
    return batch_norm_wrapper