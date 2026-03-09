import torch
import triton
import triton.language as tl

def pattern(x):
    tmp_0 = torch.nn.functional.gelu(x, approximate='none')
    tmp_1 = tmp_0.flatten(1, -1)
    return tmp_1

def replacement_args(x):
    return (x,)

@triton.jit
def gelu_flatten_kernel_dynamic(
    x_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load input data with cache hints for better performance
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    
    # Fast GELU approximation: 0.5 * x * (1.0 + tanh(sqrt(2/pi) * x))
    # This is faster than erf and often as accurate
    sqrt_2_over_pi = 0.7978845608028654  # sqrt(2/pi)
    tanh_arg = sqrt_2_over_pi * x
    tanh_val = tl.tanh(tanh_arg)
    out = 0.5 * x * (1.0 + tanh_val)
    
    # Store result with cache hints
    tl.store(out_ptr + offsets, out, mask=mask)

@triton.jit
def gelu_flatten_kernel_dynamic(
    x_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load input data
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    
    # GELU activation using erf (original formula)
    out = x * 0.5 * (1.0 + tl.erf(x / tl.sqrt(2.0)))
    
    # Store result
    tl.store(out_ptr + offsets, out, mask=mask)

@torch.fx.wrap
def gelu_flatten_optimized_dynamic(x):
    # First, flatten the input tensor from dimension 1 as expected by the pattern
    x_flattened = x.flatten(1, -1)
    n_elements = x_flattened.numel()
    
    # Use adaptive block size based on tensor size for better performance
    if n_elements < 8192:
        BLOCK_SIZE = 256
    elif n_elements < 65536:
        BLOCK_SIZE = 1024
    else:
        BLOCK_SIZE = 2048
    
    num_programs = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Create output tensor with same flattened shape
    out = torch.empty_like(x_flattened)
    
    gelu_flatten_kernel_dynamic[(num_programs,)](
        x_ptr=x_flattened,
        out_ptr=out,
        n_elements=n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out

def replacement_func():
    return gelu_flatten_optimized_dynamic