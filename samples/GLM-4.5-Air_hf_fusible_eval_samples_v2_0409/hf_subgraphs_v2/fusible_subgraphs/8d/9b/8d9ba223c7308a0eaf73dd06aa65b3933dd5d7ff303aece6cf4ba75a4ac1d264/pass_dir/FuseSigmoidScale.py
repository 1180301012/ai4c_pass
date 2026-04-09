import torch
import triton
import triton.language as tl

@triton.jit
def fused_sigmoid_scale_kernel(
    input_ptr,
    out_ptr,
    numel,
    scale: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < numel
    
    # Load input
    x = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    
    # Apply fused sigmoid and scaling: scale * sigmoid(x)
    # Sigmoid: 1 / (1 + exp(-x))
    exp_neg_x = tl.exp(-x)
    sigmoid = 1.0 / (1.0 + exp_neg_x)
    result = scale * sigmoid
    
    # Store result
    tl.store(out_ptr + offsets, result, mask=mask)

@torch.fx.wrap
def fused_sigmoid_scale(input_tensor, scale):
    numel = input_tensor.numel()
    BLOCK_SIZE = 1024
    num_programs = (numel + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    output = torch.empty_like(input_tensor)
    
    fused_sigmoid_scale_kernel[(num_programs,)](
        input_ptr=input_tensor,
        out_ptr=output,
        numel=numel,
        scale=scale,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return output

def pattern(input_tensor, sigmoid_out, scalar_mult_out, scale):
    # Pattern matching: torch.sigmoid(input_tensor) followed by scale * sigmoid_out
    tmp_9 = torch.sigmoid(input_tensor)
    tmp_10 = scale * tmp_9
    return tmp_10

def replacement_args(input_tensor, scale):
    return (input_tensor, scale)

def replacement_func():
    return fused_sigmoid_scale