import torch
import triton
import triton.language as tl

def pattern(in_4, in_0, in_1, in_2, in_3):
    """
    Matches ReLU followed by BatchNorm with training=False, momentum=0.1, eps=1e-05.
    """
    # ReLU activation
    relu_out = torch.nn.functional.relu(in_4, inplace=False)
    # BatchNorm with given parameters
    batchnorm_out = torch.nn.functional.batch_norm(
        relu_out, in_0, in_1, in_3, in_2, 
        training=False, momentum=0.1, eps=1e-05
    )
    return batchnorm_out

def replacement_args(in_4, in_0, in_1, in_2, in_3):
    """
    Extracts the necessary arguments from matched nodes.
    """
    return (in_4, in_0, in_1, in_2, in_3)

@triton.jit
def fused_relu_batchnorm_kernel(
    input_ptr,
    running_mean_ptr,
    running_var_ptr,
    weight_ptr,
    bias_ptr,
    output_ptr,
    N,
    BLOCK_SIZE: tl.constexpr,
    EPS: float = 1e-05,
    MOMENTUM: float = 0.1,
):
    # Calculate the block index and offsets
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N

    # Load data from input tensors
    input = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    running_mean = tl.load(running_mean_ptr + offsets, mask=mask, other=0.0)
    running_var = tl.load(running_var_ptr + offsets, mask=mask, other=0.0)
    weight = tl.load(weight_ptr + offsets, mask=mask, other=0.0)
    bias = tl.load(bias_ptr + offsets, mask=mask, other=0.0)

    # Apply ReLU
    relu_input = tl.where(input > 0, input, 0.0)

    # Perform the BatchNorm calculation
    normalized = (relu_input - running_mean) / tl.sqrt(running_var + EPS)
    out = normalized * weight + bias

    # Store result
    tl.store(output_ptr + offsets, out, mask=mask)

@torch.fx.wrap
def fused_relu_batchnorm(x, running_mean, running_var, weight, bias):
    N = x.numel()
    BLOCK_SIZE = 1024
    num_programs = (N + BLOCK_SIZE - 1) // BLOCK_SIZE

    output = torch.empty_like(x)
    
    fused_relu_batchnorm_kernel[
        (num_programs, )
    ](
        x_ptr=x,
        running_mean_ptr=running_mean,
        running_var_ptr=running_var,
        weight_ptr=weight,
        bias_ptr=bias,
        output_ptr=output,
        N=N,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return output

def replacement_func():
    return fused_relu_batchnorm