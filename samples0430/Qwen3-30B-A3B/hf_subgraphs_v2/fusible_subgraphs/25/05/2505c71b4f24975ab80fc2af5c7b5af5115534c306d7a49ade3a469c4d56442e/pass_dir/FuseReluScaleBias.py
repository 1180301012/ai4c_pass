import torch
import triton
import triton.language as tl

def pattern(in_2, in_1, in_0):
    tmp_2 = torch.nn.functional.relu(in_2, inplace=False)
    tmp_3 = in_1 * tmp_2
    tmp_4 = tmp_3 + in_0
    return tmp_4

def replacement_args(in_2, in_1, in_0):
    return (in_2, in_1, in_0)

@triton.jit
def fused_relu_scale_bias_kernel(
    input_ptr,
    scale_ptr,
    bias_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr
):
    block_id = tl.program_id(0)
    block_start = block_id * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    x = tl.load(input_ptr + offsets, mask=mask)
    scale = tl.load(scale_ptr)
    bias = tl.load(bias_ptr)
    
    x = tl.where(x > 0, x, 0)
    x = x * scale + bias
    
    tl.store(output_ptr + offsets, x, mask=mask)

@torch.fx.wrap
def fused_relu_scale_bias_wrapper(x, scale, bias):
    n_elements = x.numel()
    BLOCK_SIZE = 1024
    num_blocks = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    output = torch.empty_like(x)
    
    fused_relu_scale_bias_kernel[(num_blocks,)](
        x, scale, bias, output, n_elements, BLOCK_SIZE
    )
    
    return output

def replacement_func():
    return fused_relu_scale_bias_wrapper