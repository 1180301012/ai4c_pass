import torch
import triton
import triton.language as tl

def pattern(x_0, x_1, x_2):
    # ReLU activation on input
    tmp_2 = torch.nn.functional.relu(x_2, inplace=False)
    # Scale multiplication
    tmp_3 = x_1 * tmp_2
    # Bias addition - this is the target operation
    result = tmp_3 + x_0
    return result

def replacement_args(x_0, x_1, x_2):
    return (x_0, x_1, x_2)

@triton.jit
def fused_relu_scale_bias_kernel(
    input_ptr,
    scale_ptr,
    bias_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load input data
    input_val = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    
    # Load parameters (scalars)
    scale = tl.load(scale_ptr)
    bias = tl.load(bias_ptr)
    
    # Fuse all operations: ReLU -> Scale -> Bias
    # ReLU: max(0, x)
    relu_out = tl.maximum(input_val, 0.0)
    # Scale: scale * x
    scale_out = scale * relu_out
    # Bias: scale_out + bias
    bias_out = scale_out + bias
    
    # Store result
    tl.store(output_ptr + offsets, bias_out, mask=mask)

@torch.fx.wrap
def fused_relu_scale_bias(x_0, x_1, x_2):
    n_elements = x_2.numel()
    BLOCK_SIZE = 1024
    num_programs = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    output = torch.empty_like(x_2)
    
    fused_relu_scale_bias_kernel[(num_programs,)](
        input_ptr=x_2,
        scale_ptr=x_1,
        bias_ptr=x_0,
        output_ptr=output,
        n_elements=n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return output

def replacement_func():
    return fused_relu_scale_bias