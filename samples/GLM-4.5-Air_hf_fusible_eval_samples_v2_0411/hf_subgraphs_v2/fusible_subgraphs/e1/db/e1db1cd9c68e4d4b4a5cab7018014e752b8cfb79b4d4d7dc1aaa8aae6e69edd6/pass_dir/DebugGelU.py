import torch
import triton
import triton.language as tl

# Simple GELU pattern for debugging
def pattern(in_0):
    return torch.nn.functional.gelu(in_0)

def replacement_args(in_0):
    return (in_0,)

# Simple Triton GELU kernel
@triton.jit
def simple_gelu_kernel(x_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    
    # Cast to float32 for math operations (required by Triton)
    x_float = tl.cast(x, tl.float32)
    
    # Simplified GELU using sigmoid approximation: GELU(x) ≈ x * sigmoid(1.702 * x)
    sigmoid_x = 1.0 / (1.0 + tl.exp(-1.702 * x_float))
    gelu_out_float = x_float * sigmoid_x
    
    # Cast back to original dtype for storage
    gelu_out = tl.cast(gelu_out_float, tl.float32 if x_float.dtype == tl.float32 else tl.bfloat16 if x_float.dtype == tl.bfloat16 else tl.float16)
    
    tl.store(out_ptr + offsets, gelu_out, mask=mask)

# Optimized GELU kernel with fixed configuration
@triton.jit
def simple_gelu_kernel(x_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    
    # Cast to float32 for math operations (required by Triton)
    x_float = tl.cast(x, tl.float32)
    
    # Simplified GELU using sigmoid approximation: GELU(x) ≈ x * sigmoid(1.702 * x)
    sigmoid_x = 1.0 / (1.0 + tl.exp(-1.702 * x_float))
    gelu_out_float = x_float * sigmoid_x
    
    # Cast back to original dtype for storage
    gelu_out = tl.cast(gelu_out_float, tl.float32 if x_float.dtype == tl.float32 else tl.bfloat16 if x_float.dtype == tl.bfloat16 else tl.float16)
    
    tl.store(out_ptr + offsets, gelu_out, mask=mask)

@torch.fx.wrap
def simple_gelu_wrapper(x):
    n_elements = x.numel()
    
    # Choose block size based on tensor size for better utilization
    if n_elements < 1024:
        block_size = 128
    elif n_elements < 10000:
        block_size = 256
    elif n_elements < 100000:
        block_size = 512
    else:
        block_size = 1024
    
    n_programs = (n_elements + block_size - 1) // block_size
    
    output = torch.empty_like(x)
    simple_gelu_kernel[(n_programs,)](x, output, n_elements, block_size)
    
    return output

def replacement_func():
    return simple_gelu_wrapper