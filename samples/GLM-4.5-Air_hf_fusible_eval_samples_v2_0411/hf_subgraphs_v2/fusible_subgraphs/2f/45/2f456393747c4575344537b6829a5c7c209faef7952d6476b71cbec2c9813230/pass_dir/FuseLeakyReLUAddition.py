import torch
import triton
import triton.language as tl

def pattern(batch_norm_output, residual_input):
    leaky_relu = torch.nn.functional.leaky_relu(batch_norm_output, 0.01, True)
    result = leaky_relu + residual_input
    return result

def replacement_args(batch_norm_output, residual_input):
    return (batch_norm_output, residual_input)

@triton.jit
def fused_leaky_relu_add_kernel(
    x_ptr,
    y_ptr,
    out_ptr,
    n_elements,
    negative_slope: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Fused LeakyReLU + Addition kernel"""
    # Each program handles a contiguous block of data
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load tensors
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    y = tl.load(y_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    
    # Apply LeakyReLU: x = max(negative_slope * x, x)
    x = tl.where(x > 0, x, negative_slope * x)
    
    # Add the residual connection
    out = x + y
    
    # Store result
    tl.store(out_ptr + offsets, out, mask=mask)

@torch.fx.wrap
def fused_leaky_relu_add(x, y, negative_slope=0.01):
    """Fused LeakyReLU + Addition function"""
    # Ensure tensors are on the same device and have the same shape
    if x.device != y.device:
        y = y.to(x.device)
    
    # Check tensor shapes match
    if x.shape != y.shape:
        raise ValueError(f"Shape mismatch: x {x.shape} vs y {y.shape}")
    
    N = x.numel()
    
    # Use optimal block size for GPU
    BLOCK_SIZE = 1024
    
    # Calculate number of programs needed
    num_programs = (N + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Create output tensor with same properties as input
    out = torch.empty_like(x)
    
    # Launch the fused kernel
    fused_leaky_relu_add_kernel[(num_programs,)](
        x_ptr=x,
        y_ptr=y,
        out_ptr=out,
        n_elements=N,
        negative_slope=negative_slope,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out

def replacement_func():
    return fused_leaky_relu_add