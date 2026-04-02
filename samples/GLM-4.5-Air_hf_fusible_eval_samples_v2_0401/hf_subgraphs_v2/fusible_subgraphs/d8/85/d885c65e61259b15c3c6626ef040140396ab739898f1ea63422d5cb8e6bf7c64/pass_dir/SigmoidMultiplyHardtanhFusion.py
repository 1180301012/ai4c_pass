import torch
import triton
import triton.language as tl

# Pattern matching function - matches sigmoid + multiplication + hardtanh
def pattern(x, scale):
    tmp_3 = x.sigmoid()
    tmp_4 = scale * tmp_3
    tmp_5 = torch.nn.functional.hardtanh(tmp_4, 0.0, 6.0, False)
    return tmp_5

# Argument extraction function
def replacement_args(x, scale):
    return (x, scale)

# Triton kernel for fused sigmoid, multiply, and hardtanh operations
@triton.jit
def fused_sigmoid_multiply_hardtanh_kernel(
    x_ptr, scale_ptr, out_ptr,
    n_elements, BLOCK_SIZE: tl.constexpr
):
    # Get program ID
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load input values with masking
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    scale = tl.load(scale_ptr + offsets, mask=mask, other=1.0)
    
    # Fuse the three operations in one kernel:
    # 1. Sigmoid activation: 1/(1+exp(-x))
    # 2. Element-wise multiplication with scaling
    # 3. Hardtanh: clamp between 0 and 6
    sigmoid_val = 1.0 / (1.0 + tl.exp(-x))
    scaled_val = sigmoid_val * scale
    hardtanh_val = tl.maximum(tl.minimum(scaled_val, 6.0), 0.0)
    
    # Store the result
    tl.store(out_ptr + offsets, hardtanh_val, mask=mask)

@torch.fx.wrap
def fused_activation(x, scale):
    # Create output tensor
    output = torch.empty_like(x)
    
    # Launch the fused Triton kernel
    n_elements = x.numel()
    BLOCK_SIZE = 1024
    num_programs = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    fused_sigmoid_multiply_hardtanh_kernel[(num_programs,)](
        x,
        scale,
        output,
        n_elements,
        BLOCK_SIZE
    )
    
    return output

# Replacement function - returns the optimized function reference
def replacement_func():
    return fused_activation