import torch
import triton
import triton.language as tl

def pattern(scale, activation):
    return scale * activation

def replacement_args(scale, activation):
    return (scale, activation)

@triton.jit
def mul_kernel_optimized(
    scale_ptr,
    activation_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    scale = tl.load(scale_ptr)
    activation = tl.load(activation_ptr + offsets, mask=mask, other=0.0)
    
    # Vectorized multiplication
    out = activation * scale
    tl.store(out_ptr + offsets, out, mask=mask)

@torch.fx.wrap
def triton_mul_optimized(scale, activation):
    # Advanced block size optimization based on tensor size and GPU occupancy
    N = activation.numel()
    
    # Choose optimal block size for maximum occupancy
    if N >= 2048 * 2048:  # Very large tensors
        BLOCK_SIZE = 4096
    elif N >= 1024 * 1024:  # Large tensors
        BLOCK_SIZE = 2048
    elif N >= 128 * 1024:  # Medium tensors  
        BLOCK_SIZE = 1024
    elif N >= 16 * 1024:  # Small-medium tensors
        BLOCK_SIZE = 512
    else:  # Small tensors
        BLOCK_SIZE = 256
    
    num_programs = (N + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    out = torch.empty_like(activation)
    
    # Ensure tensors are on the same device and contiguous
    if not activation.is_contiguous():
        activation = activation.contiguous()
    if not scale.is_contiguous():
        scale = scale.contiguous()
    
    mul_kernel_optimized[(num_programs,)](
        scale_ptr=scale,
        activation_ptr=activation,
        out_ptr=out,
        n_elements=N,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out

def replacement_func():
    return triton_mul_optimized