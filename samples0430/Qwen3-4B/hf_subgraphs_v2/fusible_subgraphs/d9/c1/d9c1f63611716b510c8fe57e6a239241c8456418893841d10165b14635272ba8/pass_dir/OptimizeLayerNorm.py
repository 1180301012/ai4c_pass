import torch
import triton
import triton.language as tl


def pattern(linear, normalized_shape, weight, bias, eps):
    return torch.nn.functional.layer_norm(linear, normalized_shape, weight, bias, eps)

def replacement_args(linear, normalized_shape, weight, bias, eps):
    return (linear, normalized_shape, weight, bias, eps)

@triton.jit
def layer_norm_kernel(
    input_ptr,
    weight_ptr,
    bias_ptr,
    output_ptr,
    N,
    C,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    pass

@torch.fx.wrap
def layer_norm_kernel_wrapper(linear, normalized_shape, weight, bias, eps):
    N = linear.shape[0]
    C = normalized_shape[0]
    BLOCK_SIZE = 1024
    num_blocks = (N + BLOCK_SIZE - 1) // BLOCK_SIZE
    output = torch.empty_like(linear)
    
    layer_norm_kernel[(num_blocks,)](
        input_ptr=linear,
        weight_ptr=weight,
        bias_ptr=bias,
        output_ptr=output,
        N=N,
        C=C,
        eps=eps,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return output

def replacement_func():
    return layer_norm_kernel_wrapper