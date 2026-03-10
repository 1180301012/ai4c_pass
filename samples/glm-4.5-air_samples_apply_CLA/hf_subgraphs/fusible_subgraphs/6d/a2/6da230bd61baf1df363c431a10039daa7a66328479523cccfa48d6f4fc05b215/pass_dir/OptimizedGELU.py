import torch
import triton
import triton.language as tl

# Pattern matching function
def pattern(x):
    return torch.nn.functional.gelu(x)

# Argument extraction function
def replacement_args(x):
    return (x,)

# Optimized GELU kernel
@triton.jit
def gelu_kernel_ptr(
    output_ptr,
    input_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    x = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    
    # GELU approximation using sigmoid (erf approximation)
    # gelu(x) = x * sigmoid(1.702 * x)
    sigmoid_approx = 1.0 / (1.0 + tl.exp(-1.702 * x))
    gelu_val = x * sigmoid_approx
    
    tl.store(output_ptr + offsets, gelu_val, mask=mask)

@torch.fx.wrap
def optimized_gelu(x):
    n_elements = x.numel()
    BLOCK_SIZE = 1024
    num_programs = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    output = torch.empty_like(x, dtype=torch.float32, device=x.device)
    
    gelu_kernel_ptr[(num_programs,)](
        output_ptr=output,
        input_ptr=x,
        n_elements=n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return output

# Replacement function
def replacement_func():
    return optimized_gelu