import torch
import triton
import triton.language as tl

# Pattern matching function: division → ReLU → squaring
def pattern(x, scale):
    tmp_0 = x / scale
    tmp_1 = torch.nn.functional.relu(tmp_0)
    tmp_2 = torch.square(tmp_1)
    return tmp_2

# Argument extraction function  
def replacement_args(x, scale):
    return (x, scale)

# Optimized kernel using mathematical optimization: ReLU(x)² = (x >= 0) * x * x
@triton.jit
def fused_relu_square_kernel(
    x_ptr,
    out_ptr,
    scale,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a contiguous block of data
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load input data
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    
    # Mathematical optimization: ReLU(x/scale)² = ((x/scale) >= 0) * (x/scale) * (x/scale)
    # Which simplifies to: (x >= 0) * x * x / (scale * scale)
    # This avoids explicit ReLU and reduces computation
    x_div_scale = x / scale
    relu_out = tl.where(x_div_scale >= 0, x_div_scale, 0.0)
    square_out = relu_out * relu_out
    # Alternative mathematical optimization:
    # positive_mask = (x >= 0)
    # square_out = positive_mask * x_div_scale * x_div_scale
    
    # Store result
    tl.store(out_ptr + offsets, square_out, mask=mask)

# Kernel wrapper
@torch.fx.wrap
def fused_relu_square(x, scale):
    N = x.numel()
    BLOCK_SIZE = 1024  # Can be autotuned for better performance
    num_programs = (N + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    out = torch.empty_like(x)
    
    fused_relu_square_kernel[(num_programs,)](
        x_ptr=x,
        out_ptr=out,
        n_elements=N,
        scale=scale,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out

# Replacement function
def replacement_func():
    return fused_relu_square