import torch
import triton
import triton.language as tl

# Simple pattern test - just return input unchanged
def pattern(x):
    return (x,)

# Argument extraction function
def replacement_args(x):
    return (x,)

# Optimized Triton kernel fusing ReLU + Sigmoid
@triton.jit
def fused_relu_sigmoid_kernel(
    x_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a contiguous block of data
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load input
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    
    # Fused computation: Relu then Sigmoid
    # ReLU: max(x, 0)
    relu_out = tl.maximum(x, 0.0)
    # Sigmoid: 1 / (1 + exp(-relu_out))
    exp_neg = tl.exp(-relu_out)
    sigmoid_out = 1.0 / (1.0 + exp_neg)
    # Alternative using libm for better performance:
    # sigmoid_out = tl.libm.exp(-relu_out)
    # sigmoid_out = 1.0 / (1.0 + sigmoid_out)
    
    # Store output
    tl.store(out_ptr + offsets, sigmoid_out, mask=mask)

# Kernel wrapper (MUST be decorated with @torch.fx.wrap)
@torch.fx.wrap
def fused_relu_sigmoid(x):
    # Get tensor properties
    n_elements = x.numel()
    
    # Choose optimal block size
    BLOCK_SIZE = 1024
    
    # Calculate number of programs needed
    num_programs = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Create output tensor
    out = torch.empty_like(x)
    
    # Launch the fused kernel
    fused_relu_sigmoid_kernel[(num_programs,)](
        x_ptr=x,
        out_ptr=out,
        n_elements=n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out

# Replacement function (NO arguments, returns function reference)
def replacement_func():
    return fused_relu_sigmoid