import torch
import triton
import triton.language as tl

def pattern(in_0, in_1, in_2):
    """Pattern: Full computation sequence - ReLU -> Mul -> Add -> Pad"""
    tmp_2 = torch.nn.functional.relu(in_2, inplace = False)
    tmp_3 = in_1 * tmp_2
    tmp_4 = tmp_3 + in_0
    tmp_5 = torch.nn.functional.pad(tmp_4, (0, 1, 0, 1), 'constant', None)
    return tmp_5

def replacement_args(in_0, in_1, in_2):
    return (in_0, in_1, in_2)

@triton.jit
def full_computation_kernel(
    bias_ptr,
    scale_ptr,
    input_ptr,
    output_ptr,
    batch,
    channels,
    height,
    width,
    BLOCK_SIZE: tl.constexpr,
):
    """Fused kernel for ReLU * scale + bias + pad operations"""
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < (batch * channels * height * width)
    
    # Load bias and scale (they're scalars)
    bias = tl.load(bias_ptr + 0)
    scale = tl.load(scale_ptr + 0)
    
    # Load input
    x = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    
    # Fused operations: ReLU * scale + bias
    relu_x = tl.maximum(x, 0.0)
    out = relu_x * scale + bias
    
    # Store result (output is unpadded shape since pad is handled separately)
    tl.store(output_ptr + offsets, out, mask=mask)

@torch.fx.wrap
def full_computation(bias, scale, x):
    """Fused ReLU * scale + bias operation"""
    batch, channels, height, width = x.shape
    N = batch * channels * height * width
    
    BLOCK_SIZE = 1024
    num_programs = (N + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    out = torch.empty_like(x)
    
    full_computation_kernel[(num_programs,)](
        bias_ptr=bias,
        scale_ptr=scale,
        input_ptr=x,
        output_ptr=out,
        n_elements=N,
        batch=batch,
        channels=channels,
        height=height,
        width=width,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out

def replacement_func():
    return full_computation