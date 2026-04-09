import torch
import triton
import triton.language as tl

# Pattern matching function - only matches max_pool2d
def pattern(x):
    return torch.nn.functional.max_pool2d(x, 2, 2, 0, 1, ceil_mode=False, return_indices=False)

# Argument extraction function  
def replacement_args(x):
    return (x,)

# Simple Triton kernel for max_pool2d
@triton.jit
def max_pool2d_kernel(
    x_ptr,
    out_ptr,
    batch_size, channels, height, width,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    n_elements = batch_size * channels * height * width // 4  # Output is half size
    
    batch_offset = pid * BLOCK_SIZE
    offsets = batch_offset + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    if tl.any(mask):
        offset = offsets[0]
        b = offset // (channels * height // 2 * width // 2)
        c = (offset // (height // 2 * width // 2)) % channels
        h = (offset // (width // 2)) % (height // 2)
        w = offset % (width // 2)
        
        # Load 2x2 block and take max
        val00 = tl.load(x_ptr + (b * channels * height * width + c * height * width + (h*2) * width + w*2), mask=None)
        val01 = tl.load(x_ptr + (b * channels * height * width + c * height * width + (h*2) * width + w*2 + 1), mask=None)
        val10 = tl.load(x_ptr + (b * channels * height * width + c * height * width + (h*2+1) * width + w*2), mask=None)
        val11 = tl.load(x_ptr + (b * channels * height * width + c * height * width + (h*2+1) * width + w*2 + 1), mask=None)
        
        max_val = tl.maximum(tl.maximum(val00, val01), tl.maximum(val10, val11))
        
        tl.store(out_ptr + offset, max_val, mask=mask)

# Wrapper
@torch.fx.wrap
def simple_max_pool2d(x):
    batch_size, channels, height, width = x.shape
    output = torch.empty((batch_size, channels, height // 2, width // 2), dtype=x.dtype, device=x.device)
    
    total_elements = batch_size * channels * height * width // 4
    BLOCK_SIZE = 1024
    grid_size = (triton.cdiv(total_elements, BLOCK_SIZE),)
    
    max_pool2d_kernel[grid_size](
        x, output,
        batch_size, channels, height, width,
        BLOCK_SIZE
    )
    
    return output

# Replacement function
def replacement_func():
    return simple_max_pool2d