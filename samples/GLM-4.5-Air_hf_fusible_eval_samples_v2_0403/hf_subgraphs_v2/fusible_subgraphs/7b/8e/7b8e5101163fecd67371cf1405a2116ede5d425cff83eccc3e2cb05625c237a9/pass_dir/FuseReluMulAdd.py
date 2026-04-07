import torch
import triton
import triton.language as tl

def pattern(in_0, in_1, in_2):
    """Pattern: ReLU -> Multiply -> Add fusion"""
    tmp_2 = torch.nn.functional.relu(in_2, inplace = False)
    tmp_3 = in_1 * tmp_2
    tmp_4 = tmp_3 + in_0
    return tmp_4

def replacement_args(in_0, in_1, in_2):
    return (in_0, in_1, in_2)

@triton.jit
def fused_relu_mul_add_kernel(
    bias_ptr,
    scale_ptr,
    input_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """Fused ReLU * scale + bias kernel"""
    # Each program handles a contiguous block of data
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load bias and scale (they're scalars, so we just need one element)
    bias = tl.load(bias_ptr + 0)
    scale = tl.load(scale_ptr + 0)
    
    # Load input
    x = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    
    # Fused operations: ReLU * scale + bias
    relu_x = tl.maximum(x, 0.0)
    out = relu_x * scale + bias
    
    # Store result
    tl.store(output_ptr + offsets, out, mask=mask)

@torch.fx.wrap
def fused_relu_mul_add(bias, scale, x):
    """Fused ReLU * scale + bias operation"""
    N = x.numel()
    BLOCK_SIZE = 1024
    num_programs = (N + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    out = torch.empty_like(x)
    
    fused_relu_mul_add_kernel[(num_programs,)](
        bias_ptr=bias,
        scale_ptr=scale,
        input_ptr=x,
        output_ptr=out,
        n_elements=N,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out

def replacement_func():
    return fused_relu_mul_add