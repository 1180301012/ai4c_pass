import torch
import triton
import triton.language as tl

# Pattern matching function - must match the exact operations from model.py
def pattern(x):
    """
    Matches ReLU + Sigmoid fusion pattern
    Pattern: input -> ReLU(inplace=True) -> Sigmoid -> return
    """
    tmp_0 = torch.nn.functional.relu(x, inplace=True)
    tmp_1 = torch.sigmoid(tmp_0)
    return tmp_1

# Argument extraction function
def replacement_args(x):
    return (x,)

# Optimized fused kernel that combines ReLU and Sigmoid
@triton.jit
def fused_relu_sigmoid_kernel(x_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    """
    Fused ReLU + Sigmoid kernel for better performance
    Combines two activation functions into a single kernel launch
    """
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load input data
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    
    # Fused computation: ReLU then Sigmoid
    # ReLU: max(0, x)
    relu_out = tl.maximum(x, 0.0)
    # Sigmoid: 1 / (1 + exp(-relu_out))
    out = 1.0 / (1.0 + tl.exp(-relu_out))
    
    # Store result
    tl.store(out_ptr + offsets, out, mask=mask)

# Kernel wrapper that handles different data types
@torch.fx.wrap
def fused_relu_sigmoid(x):
    """
    Wrapper function that handles different data types and launches the fused kernel
    """
    n_elements = x.numel()
    
    # Choose optimal block size based on tensor size
    if n_elements < 1024:
        BLOCK_SIZE = 128
    elif n_elements < 100000:
        BLOCK_SIZE = 512
    else:
        BLOCK_SIZE = 1024
    
    num_programs = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Create output tensor with same dtype as input
    out = torch.empty_like(x)
    
    # Launch the fused kernel
    fused_relu_sigmoid_kernel[(num_programs,)](
        x_ptr=x,
        out_ptr=out,
        n_elements=n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out

# Replacement function - returns a zero-argument function that returns the kernel wrapper
def replacement_func():
    return fused_relu_sigmoid