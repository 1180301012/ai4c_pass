import torch
import triton
import triton.language as tl

def pattern(in_0):
    tmp_1 = torch.sigmoid(in_0)
    return tmp_1

def replacement_args(in_0):
    return (in_0,)

@triton.jit
def fused_relu_sigmoid_kernel(
    x_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load input
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    
    # Apply fused ReLU + Sigmoid
    # ReLU: max(0, x)
    # Sigmoid: 1 / (1 + exp(-x))
    # Combined: 1 / (1 + exp(-max(0, x)))
    
    # First apply ReLU
    relu_x = tl.maximum(x, 0.0)
    
    # Apply Sigmoid to the ReLU output
    # Using exp approximation for better performance
    sigmoid_x = 1.0 / (1.0 + tl.exp(-relu_x))
    
    # Store output
    tl.store(out_ptr + offsets, sigmoid_x, mask=mask)

@torch.fx.wrap  
def fused_relu_sigmoid(x):
    """Fused ReLU+Sigmoid activation function using Triton"""
    # Handle different data types
    if x.dtype == torch.float16:
        # Use float32 computation for better precision, then cast back
        x = x.to(torch.float32)
        output_dtype = torch.float16
    else:
        output_dtype = x.dtype
    
    N = x.numel()
    
    # Choose block size based on tensor size for better occupancy
    if N < 8192:
        BLOCK_SIZE = 128
    elif N < 65536:
        BLOCK_SIZE = 256
    elif N < 524288:
        BLOCK_SIZE = 1024
    else:
        BLOCK_SIZE = 2048
    
    num_programs = (N + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Create output tensor
    out = torch.empty_like(x, dtype=output_dtype)
    
    # Launch kernel
    fused_relu_sigmoid_kernel[num_programs,](
        x_ptr=x,
        out_ptr=out,
        n_elements=N,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    # Cast back to original dtype if needed
    if x.dtype == torch.float16 and output_dtype == torch.float16:
        pass  # already correct
    elif x.dtype == torch.float16 and out.dtype == torch.float32:
        out = out.to(torch.float16)
    
    return out

def replacement_func():
    return fused_relu_sigmoid