import torch
import triton
import triton.language as tl


def pattern(in_0):
    """Match ReLU followed by Sigmoid pattern."""
    tmp = torch.relu(in_0)
    out = torch.sigmoid(tmp)
    return out


def replacement_args(in_0):
    return (in_0,)


@triton.jit
def relu_sigmoid_kernel(
    input_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """Fused ReLU + Sigmoid kernel."""
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load input
    x = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    
    # ReLU: clamp to [0, inf)
    x_relu = tl.where(x > 0, x, 0.0)
    
    # Sigmoid: 1 / (1 + exp(-x))
    neg_x = -x_relu
    neg_x = tl.where(neg_x > 20.0, 20.0, neg_x)
    exp_neg_x = tl.exp(neg_x)
    result = 1.0 / (1.0 + exp_neg_x)
    
    # Store result
    tl.store(output_ptr + offsets, result, mask=mask)


@torch.fx.wrap
def relu_sigmoid_fused(x):
    """Fused ReLU + Sigmoid kernel wrapper."""
    n_elements = x.numel()
    BLOCK_SIZE = 1024
    num_programs = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    if x.dim() > 1:
        x_flat = x.flatten()
        output = torch.empty_like(x_flat)
    else:
        x_flat = x
        output = torch.empty_like(x)
    
    relu_sigmoid_kernel[(num_programs,)](
        input_ptr=x_flat,
        output_ptr=output,
        n_elements=n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    if x.dim() > 1:
        output = output.reshape(x.shape)
    
    return output


def replacement_func():
    return relu_sigmoid_fused