import torch
import triton
import triton.language as tl

# Pattern matching function for in_2.sigmoid()
def pattern(in_2):
    return in_2.sigmoid()

# Argument extraction function
def replacement_args(in_2):
    return (in_2,)

# Sigmoid kernel for element-wise sigmoid operation
@triton.jit
def sigmoid_kernel(
    x_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # Create mask for elements within bounds
    mask = offsets < n_elements
    
    # Load input data
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    
    # Compute sigmoid: 1 / (1 + exp(-x))
    # Fast and accurate sigmoid approximation
    neg_x = -x
    exp_val = tl.exp(neg_x)
    out = 1.0 / (1.0 + exp_val)
    
    # Store result
    tl.store(out_ptr + offsets, out, mask=mask)

# Wrapper function
@torch.fx.wrap
def triton_sigmoid(x):
    n_elements = x.numel()
    
    # For small tensors, use smaller block size to reduce overhead
    # 76,800 elements is relatively small
    BLOCK_SIZE = 256
    num_programs = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Create output tensor
    out = torch.empty_like(x, dtype=torch.float32)
    
    # Launch kernel
    sigmoid_kernel[(num_programs,)](
        x_ptr=x,
        out_ptr=out,
        n_elements=n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out

# Replacement function
def replacement_func():
    return triton_sigmoid