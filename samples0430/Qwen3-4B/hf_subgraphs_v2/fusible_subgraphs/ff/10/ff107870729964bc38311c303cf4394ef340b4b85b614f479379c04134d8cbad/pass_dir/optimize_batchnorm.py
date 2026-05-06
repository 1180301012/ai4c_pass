import torch
import triton
import triton.language as tl

def pattern(input, running_mean, running_var, weight, bias):
    return torch.nn.functional.batch_norm(
        input,
        running_mean,
        running_var,
        weight,
        bias,
        training=False,
        momentum=0.1,
        eps=1e-5
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
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    # Get the current program ID and compute the block start
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    # Define offsets for the current block
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    # Load data
    input_vals = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    input_vals = tl.cast(input_vals, tl.float32)
    input_vals = tl.cast(input_vals, tl.float32)
    running_mean_vals = tl.load(running_mean_ptr + offsets, mask=mask, other=0.0)
    running_mean_vals = tl.cast(running_mean_vals, tl.float32)
    running_var_vals = tl.load(running_var_ptr + offsets, mask=mask, other=0.0)
    running_var_vals = tl.cast(running_var_vals, tl.float32)
    weight_vals = tl.load(weight_ptr + offsets, mask=mask, other=0.0)
    weight_vals = tl.cast(weight_vals, tl.float32)
    bias_vals = tl.load(bias_ptr + offsets, mask=mask, other=0.0)
    bias_vals = tl.cast(bias_vals, tl.float32)
    # Calculate the normalized values
    mean_sub = input_vals - running_mean_vals
    var_add = running_var_vals + 1e-5
    sqrt_var = tl.sqrt(var_add)
    norm = mean_sub / sqrt_var
    output_vals = norm * weight_vals + bias_vals
    # Store results
    tl.store(output_ptr + offsets, output_vals, mask=mask)

@torch.fx.wrap
def batchnorm_wrapper(input, running_mean, running_var, weight, bias):
    N = input.numel()
    BLOCK_SIZE = 1024
    num_programs = (N + BLOCK_SIZE - 1) // BLOCK_SIZE
    output = torch.empty_like(input)
    batchnorm_kernel[(num_programs,)](
        input_ptr=input,
        running_mean_ptr=running_mean,
        running_var_ptr=running_var,
        weight_ptr=weight,
        bias_ptr=bias,
        output_ptr=output,
        n_elements=N,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return output

def replacement_func():
    return batchnorm_wrapper