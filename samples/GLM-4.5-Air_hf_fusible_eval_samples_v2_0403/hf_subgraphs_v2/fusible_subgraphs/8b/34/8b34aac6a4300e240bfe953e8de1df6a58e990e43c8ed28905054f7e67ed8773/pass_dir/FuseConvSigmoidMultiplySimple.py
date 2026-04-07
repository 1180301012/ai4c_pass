import torch
import triton
import triton.language as tl

def pattern(x, y, z):
    """
    Pattern to match sigmoid -> multiply sequence with dummy input
    This matches the logical computation unit in both branches
    """
    # Fused sigmoid and multiply operations
    a = torch.sigmoid(x)
    b = a * y
    # Use z parameter to avoid dead code error
    result = b + z * 0.0  # z * 0.0 doesn't change the result
    return result

def replacement_args(x, y, z):
    return (x, y, z)

@triton.jit
def simple_fused_kernel(
    x_ptr, y_ptr, z_ptr, output_ptr,
    n_elements: tl.constexpr,
):
    """
    Simple fused kernel for demonstration
    """
    pid = tl.program_id(0)
    block_start = pid * 1024
    offsets = block_start + tl.arange(0, 1024)
    mask = offsets < n_elements
    
    # Load inputs
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = tl.load(y_ptr + offsets, mask=mask, other=0.0)
    z = tl.load(z_ptr + offsets, mask=mask, other=0.0)
    
    # Fused computation: sigmoid(x) * y + z * 0.0
    sigmoid_x = 1.0 / (1.0 + tl.exp(-x))
    result = sigmoid_x * y + z * 0.0
    
    # Store result
    tl.store(output_ptr + offsets, result, mask=mask)

@torch.fx.wrap
def simple_fused_function(x, y, z):
    """Simple wrapper for testing pattern matching"""
    # Flatten inputs for simple kernel
    x_flat = x.flatten()
    y_flat = y.flatten()
    z_flat = z.flatten()
    
    n_elements = x_flat.numel()
    output = torch.empty_like(x_flat)
    
    # Launch kernel
    grid_size = (n_elements + 1023) // 1024
    simple_fused_kernel[grid_size](
        x_flat, y_flat, z_flat, output,
        n_elements
    )
    
    # Return in original shape
    return output.reshape(x.shape)

def replacement_func():
    return simple_fused_function