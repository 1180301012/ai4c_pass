import torch
import triton
import triton.language as tl

def pattern(x):
    # Simple sigmoid pattern to test basic matching
    return torch.sigmoid(x)

def replacement_args(x):
    return (x,)

@triton.jit
def fused_relu_sigmoid_kernel(
    x_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a contiguous block of data
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load input data
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    
    # Fused ReLU + Sigmoid computation
    # ReLU: max(0, x)
    relu_out = tl.maximum(x, 0.0)
    # Sigmoid: 1 / (1 + exp(-relu_out))
    sigmoid_out = 1.0 / (1.0 + tl.exp(-relu_out))
    
    # Store result
    tl.store(out_ptr + offsets, sigmoid_out, mask=mask)

@torch.fx.wrap
def fused_relu_sigmoid(x):
    # Get input tensor info
    N = x.numel()
    
    # Choose block size based on tensor characteristics
    if N <= 1024:
        BLOCK_SIZE = 128
    elif N <= 16384:
        BLOCK_SIZE = 256
    elif N <= 65536:
        BLOCK_SIZE = 512
    else:
        BLOCK_SIZE = 1024
    
    # Calculate number of programs needed
    num_programs = (N + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Create output tensor
    out = torch.empty_like(x)
    
    # Launch the fused kernel
    fused_relu_sigmoid_kernel[(num_programs,)](
        x_ptr=x,
        out_ptr=out,
        n_elements=N,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out

def replacement_func():
    return fused_relu_sigmoid