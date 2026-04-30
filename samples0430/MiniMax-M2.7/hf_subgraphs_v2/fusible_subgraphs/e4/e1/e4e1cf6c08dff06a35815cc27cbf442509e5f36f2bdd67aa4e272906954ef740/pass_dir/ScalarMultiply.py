import torch
import triton
import triton.language as tl

# Pattern matching function for scalar multiplication
def pattern(in_0, in_2):
    """
    Pattern matching the scalar multiplication.
    """
    return in_0 * in_2

# Argument extraction function
def replacement_args(in_0, in_2):
    return (in_0, in_2)

@triton.jit
def scalar_mul_kernel(
    x_ptr,
    scalar_ptr,
    out_ptr,
    n_elements: tl.int32,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Scalar multiplication kernel.
    
    Each program handles a contiguous block of data.
    """
    pid = tl.program_id(0)
    block_size_offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = block_size_offsets < n_elements
    
    # Load input
    x = tl.load(x_ptr + block_size_offsets, mask=mask, other=0.0)
    
    # Load scalar (broadcast)
    scalar = tl.load(scalar_ptr)
    
    # Multiply
    out = x * scalar
    
    # Store
    tl.store(out_ptr + block_size_offsets, out, mask=mask)

# Autotune configurations
autotune_configs = [
    triton.Config({'BLOCK_SIZE': 1024}, num_stages=1, num_warps=4),
    triton.Config({'BLOCK_SIZE': 2048}, num_stages=1, num_warps=8),
]

@torch.fx.wrap
def scalar_mul_wrapper(in_0, in_2):
    """
    Wrapper for the scalar multiplication kernel.
    
    Args:
        in_0: Input tensor [B, S, H]
        in_2: Scalar tensor (0-dimensional)
    
    Returns:
        scalar * input
    """
    n_elements = in_0.numel()
    BLOCK_SIZE = 1024
    num_programs = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Move scalar to GPU if needed
    if in_2.device.type == 'cpu':
        scalar_gpu = in_2.to(device=in_0.device, dtype=torch.float32)
    else:
        scalar_gpu = in_2.to(dtype=torch.float32)
    
    # Create output tensor
    out = torch.empty_like(in_0)
    
    # Grid configuration
    grid = (num_programs,)
    
    # Launch kernel with autotuning
    scalar_mul_kernel_autotuned[grid](
        x_ptr=in_0,
        scalar_ptr=scalar_gpu,
        out_ptr=out,
        n_elements=n_elements,
    )
    
    return out

# Autotuned kernel
scalar_mul_kernel_autotuned = triton.autotune(
    configs=autotune_configs,
    key=['n_elements'],
)(scalar_mul_kernel)

def replacement_func():
    return scalar_mul_wrapper