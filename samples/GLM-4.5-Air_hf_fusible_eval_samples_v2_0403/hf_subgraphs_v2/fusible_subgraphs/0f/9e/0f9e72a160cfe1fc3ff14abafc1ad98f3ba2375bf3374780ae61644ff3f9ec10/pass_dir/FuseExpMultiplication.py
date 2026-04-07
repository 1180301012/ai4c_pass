import torch
import triton
import triton.language as tl

def pattern(x, y):
    exp_x = x.exp()
    result = exp_x * y
    return result

def replacement_args(x, y):
    return (x, y)

@triton.jit
def exp_mul_kernel(
    x_ptr,
    y_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a contiguous block of data
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # Create mask to handle elements beyond tensor size
    mask = offsets < n_elements
    
    # Load values with masking - convert to fp32 for exp computation
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    y = tl.load(y_ptr + offsets, mask=mask, other=0.0)
    
    # Compute exp(x) * y
    # Use fast approximate exp for better performance
    exp_x = tl.exp(x)
    out = exp_x * y.to(tl.float32)
    
    # Convert back to original dtype and store result
    tl.store(out_ptr + offsets, out.to(y.dtype), mask=mask)

@torch.fx.wrap
def fused_exp_mul(x, y):
    # Handle broadcasting
    # If x is scalar (like in our case), we need to broadcast it to match y's shape
    if x.numel() == 1:
        x_expanded = x.expand_as(y)
    else:
        x_expanded = x
    
    # For small tensors, use PyTorch's optimized operations instead of Triton
    # This avoids compilation issues and is often faster for small tensors
    if y.numel() <= 2048:  # Small threshold for PyTorch optimization
        exp_x = x_expanded.exp()
        return exp_x * y
    else:
        # Create output tensor
        out = torch.empty_like(y)
        N = y.numel()
        
        # Set Triton kernel configuration
        BLOCK_SIZE = 1024
        grid = (triton.cdiv(N, BLOCK_SIZE),)
        
        # Launch kernel
        exp_mul_kernel[grid](
            x_expanded,
            y,
            out,
            N,
            BLOCK_SIZE=BLOCK_SIZE,
        )
        
        return out

def replacement_func():
    return fused_exp_mul