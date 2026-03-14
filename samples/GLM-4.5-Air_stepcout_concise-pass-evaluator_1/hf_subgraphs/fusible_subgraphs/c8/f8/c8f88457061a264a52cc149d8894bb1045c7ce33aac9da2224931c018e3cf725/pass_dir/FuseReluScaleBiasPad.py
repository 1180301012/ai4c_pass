import torch
import triton
import triton.language as tl

def pattern(x, y):
    return x + y

def replacement_args(x, y):
    return (x, y)

@triton.jit
def fused_relu_scale_bias_kernel(
    input_ptr,
    bias_ptr,
    scale_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a contiguous block of data
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load input data
    input = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    
    # Load bias and scale (they are scalars, broadcast them)
    bias = tl.load(bias_ptr)
    scale = tl.load(scale_ptr)
    
    # Fuse all operations: ReLU -> Scale -> Bias
    # ReLU: max(0, x)
    relu_out = tl.maximum(input, 0.0)
    # Scale: scale * x
    scale_out = scale * relu_out
    # Bias: scale_out + bias
    bias_out = scale_out + bias
    
    # Store result
    tl.store(output_ptr + offsets, bias_out, mask=mask)

@triton.jit
def triton_add_kernel(
    x_ptr,
    y_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a contiguous block of data of size BLOCK_SIZE
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements # Mask to ensure we don't go out of bounds
    
    # Load x (the larger tensor)
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    
    # Load y (the scalar bias) - it will be broadcast to all elements
    y = tl.load(y_ptr)
    
    # Calculate with broadcasting
    out = x + y
    
    # Store
    tl.store(out_ptr + offsets, out, mask=mask)

@torch.fx.wrap
def triton_add(x, y):
    N = x.numel()
    BLOCK_SIZE = 1024
    num_programs = (N + BLOCK_SIZE - 1) // BLOCK_SIZE

    out = torch.empty_like(x)

    triton_add_kernel[(num_programs,)](
        x_ptr=x,
        y_ptr=y,
        out_ptr=out,
        n_elements=N,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    return out

def replacement_func():
    return triton_add