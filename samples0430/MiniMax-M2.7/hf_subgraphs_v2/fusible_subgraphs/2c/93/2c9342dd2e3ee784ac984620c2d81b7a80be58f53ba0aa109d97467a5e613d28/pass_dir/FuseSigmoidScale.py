import torch
import triton
import triton.language as tl

@triton.jit
def sigmoid_scale_kernel(
    x_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fused sigmoid-scale kernel.
    Computes: (sigmoid(x) - 0.25) * π
    """
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load input - let Triton handle dtype
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    
    # Convert to float32 for exp (required for bf16/fp16)
    x_fp32 = x.to(tl.float32)
    
    # Sigmoid: 1 / (1 + exp(-x))
    x_sigmoid = 1.0 / (1.0 + tl.exp(-x_fp32))
    
    # Fused scale and bias: (sigmoid - 0.25) * π
    PI = 3.141592653589793
    out = (x_sigmoid - 0.25) * PI
    
    # Store with same dtype as input
    tl.store(output_ptr + offsets, out.to(x.type), mask=mask)


@torch.fx.wrap
def sigmoid_scale_wrapper(x):
    """
    Fused sigmoid-scale wrapper.
    """
    n_elements = x.numel()
    BLOCK_SIZE = 512
    num_programs = triton.cdiv(n_elements, BLOCK_SIZE)
    
    output = torch.empty_like(x)
    
    sigmoid_scale_kernel[(num_programs,)](
        x_ptr=x,
        output_ptr=output,
        n_elements=n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return output


def pattern(tmp_4):
    """
    Match the pattern: sigmoid -> subtract 0.25 -> multiply by π
    This pattern is: (sigmoid(x) - 0.25) * π
    """
    tmp_5 = tmp_4.sigmoid()
    tmp_6 = tmp_5 - 0.25
    tmp_7 = tmp_6 * 3.141592653589793
    return tmp_7


def replacement_args(tmp_4):
    return (tmp_4,)


def replacement_func():
    return sigmoid_scale_wrapper