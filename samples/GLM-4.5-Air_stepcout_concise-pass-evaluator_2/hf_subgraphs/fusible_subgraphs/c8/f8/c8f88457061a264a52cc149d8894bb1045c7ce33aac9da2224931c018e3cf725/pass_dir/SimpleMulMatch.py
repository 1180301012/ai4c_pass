import torch
import triton
import triton.language as tl

def pattern(tmp_1, tmp_2):
    """Simple multiplication pattern matching"""
    tmp_3 = tmp_1 * tmp_2
    return tmp_3

def replacement_args(tmp_1, tmp_2):
    """Extract arguments for the multiplication kernel"""
    return (tmp_1, tmp_2)

@triton.jit
def broadcast_mul_kernel(
    scalar_ptr,
    x_ptr,
    out_ptr,
    N, C, H, W,
    BLOCK_SIZE: tl.constexpr,
):
    """Broadcasting multiplication kernel for scalar * tensor"""
    pid = tl.program_id(0)
    
    offset = pid * BLOCK_SIZE
    offsets = offset + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N * C * H * W
    
    # Load scalar value
    scalar = tl.load(scalar_ptr)
    
    # Load tensor values
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    
    # Broadcasting multiplication
    mul = scalar * x
    
    tl.store(out_ptr + offsets, mul, mask=mask)

@triton.jit
def fast_broadcast_mul_kernel(
    scalar_ptr,
    x_ptr,
    out_ptr,
    N, C, H, W,
    BLOCK_SIZE: tl.constexpr,
):
    """Optimized fast broadcasting multiplication kernel"""
    pid = tl.program_id(0)
    
    # Calculate element positions
    elem_positions = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = elem_positions < N * C * H * W
    
    # Load scalar once per program
    scalar = tl.load(scalar_ptr)
    
    # Vectorized loading and computation
    x = tl.load(x_ptr + elem_positions, mask=mask, other=0.0)
    mul = scalar * x
    
    # Vectorized storing
    tl.store(out_ptr + elem_positions, mul, mask=mask)

@torch.fx.wrap
def broadcast_mul_triton(scalar, tensor):
    """Broadcasting multiplication wrapper with autotuning"""
    if tensor.dim() != 4:
        # Fallback for non-4D tensors
        return scalar * tensor
    
    N, C, H, W = tensor.shape
    total_elements = N * C * H * W
    
    # Choose kernel and block size based on tensor size
    if total_elements < 4096:  # Very small tensors
        BLOCK_SIZE = 64
        grid = ((total_elements + BLOCK_SIZE - 1) // BLOCK_SIZE,)
        kernel = fast_broadcast_mul_kernel
    elif total_elements < 16384:  # Small tensors
        BLOCK_SIZE = 256
        grid = ((total_elements + BLOCK_SIZE - 1) // BLOCK_SIZE,)
        kernel = fast_broadcast_mul_kernel
    elif total_elements < 1000000:  # Medium tensors
        BLOCK_SIZE = 1024
        grid = ((total_elements + BLOCK_SIZE - 1) // BLOCK_SIZE,)
        kernel = fast_broadcast_mul_kernel
    else:  # Large tensors
        BLOCK_SIZE = 2048
        grid = ((total_elements + BLOCK_SIZE - 1) // BLOCK_SIZE,)
        kernel = fast_broadcast_mul_kernel
    
    # Create output tensor
    out = torch.empty_like(tensor)
    
    # Launch optimized broadcasting multiplication kernel
    kernel[grid](
        scalar_ptr=scalar,
        x_ptr=tensor,
        out_ptr=out,
        N=N, C=C, H=H, W=W,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return out

def replacement_func():
    """Return the broadcasting multiplication function"""
    return broadcast_mul_triton