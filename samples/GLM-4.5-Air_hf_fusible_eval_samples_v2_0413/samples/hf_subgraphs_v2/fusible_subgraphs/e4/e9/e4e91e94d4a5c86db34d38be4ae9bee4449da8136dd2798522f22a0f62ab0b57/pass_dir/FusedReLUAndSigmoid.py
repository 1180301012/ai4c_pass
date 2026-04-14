import torch
import triton
import triton.language as tl

# Pattern matching function
def pattern(in_0):
    """
    Matches ReLU followed by Sigmoid operations
    Pattern must mirror the exact dataflow from model.py
    """
    tmp_0 = torch.nn.functional.relu(in_0, inplace=True)
    tmp_1 = torch.sigmoid(tmp_0)
    return tmp_1

# Argument extraction function
def replacement_args(in_0):
    return (in_0,)

# Triton kernel for fused ReLU + Sigmoid
@triton.jit
def fused_relu_sigmoid_kernel(
    x_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fused kernel that applies ReLU followed by Sigmoid
    y = 1 / (1 + exp(-max(x, 0)))
    """
    # Each program handles a contiguous block of data
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements  # Mask to prevent out-of-bounds access
    
    # Load input data
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    
    # Apply ReLU: max(x, 0)
    relu_out = tl.maximum(x, 0.0)
    
    # Apply Sigmoid: 1 / (1 + exp(-relu_out))
    # Using fast exponential approximation for better performance
    sigmoid_out = 1.0 / (1.0 + tl.exp(-relu_out))
    
    # Store result
    tl.store(out_ptr + offsets, sigmoid_out, mask=mask)

# Kernel wrapper
@torch.fx.wrap
def fused_relu_sigmoid_triton(x):
    """
    Triton-optimized fused ReLU + Sigmoid activation
    """
    N = x.numel()
    BLOCK_SIZE = 1024  # Optimal block size for most GPUs
    num_programs = (N + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Create output tensor with same dtype and device as input
    out = torch.empty_like(x)
    
    # Launch kernel
    fused_relu_sigmoid_kernel[(num_programs,)](
        x_ptr=x,
        out_ptr=out,
        n_elements=N,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out

# Replacement function
def replacement_func():
    return fused_relu_sigmoid_triton