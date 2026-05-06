import torch
import triton
import triton.language as tl

def pattern(in0, in1, in2, in3, in_4, in_5, in_6, in_7, in_8, in_9):
    """Matches the batch_norm + silu pattern from the computation graph.\n\nThis pattern identifies the sequence where:\n    - batch_norm is applied to a tensor\n    - silu activation function is applied to the result\n"""
    return (in_4, in_5, in_6, in_7, in_8, in_9)

def replacement_args(in0, in1, in2, in3, in_4, in_5, in_6, in_7, in_8, in_9):
    """Extracts the necessary arguments for the replacement kernel.\n"""
    return (in0, in1, in2, in3, in_4, in_5, in_6, in_7, in_8, in_9)

@triton.jit
def fused_batch_norm_silu_kernel(
    input_ptr,
    running_mean_ptr,
    running_var_ptr,
    weight_ptr,
    bias_ptr,
    eps,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """GPU kernel that fuses batch normalization with Silu activation.\n"""
    # Compute the index for the current block
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # Load input parameters
    x = tl.load(input_ptr + offsets, mask=mask, other=0.0)

    # Load fixed values (we assume these are broadcasted for batch norm)
    mean = tl.load(running_mean_ptr)
    var = tl.load(running_var_ptr)
    weight = tl.load(weight_ptr)
    bias = tl.load(bias_ptr)

    # Compute batch norm: (x - mean) / sqrt(var + eps)
    x_mean = x - mean
    x_std = tl.sqrt(var + eps)
    x_norm = x_mean / x_std

    # Apply weight and bias
    x_out = x_norm * weight + bias

    # Apply Silu activation: x * (1 + exp(-x))
    silu_out = x_out * (1.0 + tl.exp(-x_out))

    # Store result
    tl.store(input_ptr + offsets, silu_out, mask=mask)

@torch.fx.wrap
def fused_batch_norm_silu(input, running_mean, running_var, weight, bias, eps=1e-5):
    """Wrapper function for the fused kernel.\n"""
    N = input.numel()
    BLOCK_SIZE = 1024  # Good block size for this operation
    num_programs = (N + BLOCK_SIZE - 1) // BLOCK_SIZE

    out = torch.empty_like(input)

    fused_batch_norm_silu_kernel[
        (num_programs,)
    ](
        input_ptr=input,
        running_mean_ptr=running_mean,
        running_var_ptr=running_var,
        weight_ptr=weight,
        bias_ptr=bias,
        eps=eps,
        n_elements=N,
        BLOCK_SIZE=BLOCK_SIZE
    )

    return out

def replacement_func():
    """Returns the replacement function for the pass.\n"""
    return fused_batch_norm_silu