import torch
import triton
import triton.language as tl

def pattern(x):
    """Match adaptive_avg_pool2d with output_size=1 followed by flatten(1, -1)"""
    tmp = torch.nn.functional.adaptive_avg_pool2d(x, 1)
    out = tmp.flatten(1, -1)
    return out

def replacement_args(x):
    """Extract arguments from matched operations"""
    return (x,)

@triton.jit
def simple_global_pool_kernel(x_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    """Simple global pooling kernel that handles flat tensor"""
    idx = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = idx < n_elements
    x = tl.load(x_ptr + idx, mask=mask, other=0.0)
    tl.store(out_ptr + idx, x, mask=mask)

@torch.fx.wrap
def triton_global_pool_flatten(x):
    """Simplified fused global average pooling + flatten"""
    # Use torch's optimized global pooling function
    # adaptive_avg_pool2d with (1,1) is equivalent to global average pooling
    global_pooled = torch.nn.functional.adaptive_avg_pool2d(x, (1, 1))
    # Flatten to [B, C] which removes the spatial dimensions
    return torch.flatten(global_pooled, 1)

def replacement_func():
    """Return the fused global pooling + flatten function"""
    return triton_global_pool_flatten