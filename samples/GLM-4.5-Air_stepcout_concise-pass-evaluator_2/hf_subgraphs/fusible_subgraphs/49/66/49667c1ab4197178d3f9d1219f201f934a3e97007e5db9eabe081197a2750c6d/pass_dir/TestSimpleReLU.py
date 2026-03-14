import torch
import triton
import triton.language as tl

def pattern(tmp_1):
    """Simple pattern matching: adaptive_avg_pool2d"""
    tmp_2 = torch.nn.functional.adaptive_avg_pool2d(tmp_1, 1)
    return tmp_2

def replacement_args(tmp_1):
    """Extract arguments for the replacement function"""
    return tmp_1,

@triton.jit
def simple_relu_kernel(
    x_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
    relu_only: tl.constexpr = True,
):
    """Simple ReLU kernel"""
    # Each program handles a contiguous block of data
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load input
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    
    # Compute ReLU
    out = tl.maximum(x, 0.0)
    
    # Store result
    tl.store(out_ptr + offsets, out, mask=mask)

@torch.fx.wrap
def optimized_pool2d(tmp_1):
    """Optimized adaptive_avg_pool2d wrapper function"""
    # Simple but effective optimization: use torch.mean instead of adaptive_avg_pool2d
    # For size=1, adaptive_avg_pool2d is equivalent to computing mean across spatial dimensions
    output = tmp_1.mean(dim=[2, 3], keepdim=True)
    return output

@triton.jit
def fused_relu_add_kernel(
    x_ptr,
    y_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """Fused ReLU + Add kernel: max(0, x) + y"""
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = tl.load(y_ptr + offsets, mask=mask, other=0.0)
    
    relu_x = tl.maximum(x, 0.0)
    out = relu_x + y
    
    tl.store(out_ptr + offsets, out, mask=mask)

def simple_relu(in_0, in_1):
    """Simple ReLU wrapper function for backward compatibility"""
    return fused_relu_add(in_0, in_1)

def replacement_func():
    """Return the optimized pool2d function"""
    return optimized_pool2d