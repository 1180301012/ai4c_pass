import torch
import triton
import triton.language as tl

@triton.jit
def gelu_multiply_kernel(
    x_ptr,
    y_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """Fused GELU + Element-wise multiplication kernel with adaptive block sizing"""
    # Each program handles a contiguous block of data
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load both input tensors with vectorized loads if possible
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = tl.load(y_ptr + offsets, mask=mask, other=0.0)
    
    # Optimized GELU computation using sigmoid approximation
    # GELU(x) ≈ x * sigmoid(1.702 * x) where sigmoid(x) = 1 / (1 + exp(-x))
    exp_val = tl.math.exp(-1.702 * x)
    sigmoid = 1.0 / (1.0 + exp_val)
    
    gelu_x = x * sigmoid
    
    # Multiply by second tensor
    out = gelu_x * y
    
    # Store result
    tl.store(out_ptr + offsets, out, mask=mask)

@torch.fx.wrap
def fused_gelu_multiply(x, y):
    """Fused GELU + Element-wise multiplication implementation with optimized block sizes"""
    N = x.numel()
    
    # Adapt block size based on input size
    if N < 500000:
        BLOCK_SIZE = 128
    elif N < 2000000:
        BLOCK_SIZE = 256
    else:
        BLOCK_SIZE = 1024
    
    # Calculate number of programs needed
    num_programs = (N + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Allocate output tensor
    out = torch.empty_like(x)
    
    # Launch kernel
    gelu_multiply_kernel[(num_programs,)](
        x_ptr=x,
        y_ptr=y,
        out_ptr=out,
        n_elements=N,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out

def pattern(in_0, in_1):
    """Pattern matching GELU followed by element-wise multiplication"""
    tmp_0 = torch.nn.functional.gelu(in_0, approximate='none')
    tmp_1 = tmp_0 * in_1
    return tmp_1

def replacement_args(in_0, in_1):
    """Extract arguments for the replacement function"""
    return (in_0, in_1)

def replacement_func():
    """Return the fused GELU + multiplication function"""
    return fused_gelu_multiply