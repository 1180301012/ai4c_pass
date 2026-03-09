import torch
import triton
import triton.language as tl

def pattern(in_0, in_1, in_2, in_3):
    # Match the core arithmetic pattern: (in_3 + in_2) * in_1 + in_0
    # The actual computation also includes slicing, but we'll focus on the arithmetic fusion first
    tmp_2 = in_3 + in_2
    tmp_3 = tmp_2 * in_1
    result = tmp_3 + in_0
    return result

def replacement_args(in_0, in_1, in_2, in_3):
    return (in_0, in_1, in_2, in_3)

# Optimized kernel that fuses arithmetic operations with autotune
@triton.jit
def fused_arithmetic_kernel(
    in_0_ptr, in_1_ptr, in_2_ptr, in_3_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a contiguous block of data
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load inputs with proper caching
    in_0 = tl.load(in_0_ptr + offsets, mask=mask, other=0.0)
    in_1 = tl.load(in_1_ptr + offsets, mask=mask, other=0.0)
    in_2 = tl.load(in_2_ptr + offsets, mask=mask, other=0.0)
    in_3 = tl.load(in_3_ptr + offsets, mask=mask, other=0.0)
    
    # Fused computation: (in_3 + in_2) * in_1 + in_0
    # This order minimizes temporary memory usage
    result = in_0 + (in_1 * (in_2 + in_3))
    
    # Store result
    tl.store(out_ptr + offsets, result, mask=mask)

# Optimized kernel with dynamic block size selection
@triton.jit
def fused_arithmetic_kernel(
    in_0_ptr, in_1_ptr, in_2_ptr, in_3_ptr,
    out_ptr,
    n_elements,
    block_size: tl.constexpr,
):
    # Each program handles a contiguous block of data
    pid = tl.program_id(0)
    block_start = pid * block_size
    offsets = block_start + tl.arange(0, block_size)
    mask = offsets < n_elements
    
    # Load inputs with vectorized memory access
    in_0 = tl.load(in_0_ptr + offsets, mask=mask, other=0.0)
    in_1 = tl.load(in_1_ptr + offsets, mask=mask, other=0.0)
    in_2 = tl.load(in_2_ptr + offsets, mask=mask, other=0.0)
    in_3 = tl.load(in_3_ptr + offsets, mask=mask, other=0.0)
    
    # Fused computation with optimized arithmetic order
    # This order minimizes temporary operations and improves arithmetic efficiency
    result = in_0 + (in_1 * (in_2 + in_3))
    
    # Store result with coalesced memory access
    tl.store(out_ptr + offsets, result, mask=mask)

@torch.fx.wrap
def fused_arithmetic(in_0, in_1, in_2, in_3):
    # Determine output shape based on the largest input
    if hasattr(in_2, 'shape'):
        output_shape = in_2.shape
    elif hasattr(in_3, 'shape'):
        output_shape = in_3.shape
    else:
        # Fallback if inputs don't have shape attribute
        output_shape = in_0.shape
    
    # Create output tensor
    out = torch.empty(output_shape, device=in_0.device, dtype=in_0.dtype)
    
    # Calculate total elements
    n_elements = out.numel()
    
    # Dynamic block size selection based on problem size
    # For smaller tensors, use smaller blocks to reduce overhead
    # For larger tensors, use larger blocks for better occupancy
    if n_elements < 1024:
        block_size = 256
    elif n_elements < 8192:
        block_size = 512
    elif n_elements < 65536:
        block_size = 1024
    else:
        block_size = 2048
    
    # Adjust block size to be power of 2 for better hardware utilization
    block_size = 2 ** int(block_size.bit_length() - 1)
    
    # Calculate number of programs
    num_programs = (n_elements + block_size - 1) // block_size
    
    # Ensure minimum number of programs for GPU utilization
    if num_programs < 32:
        block_size = max(256, n_elements // 32)
        num_programs = (n_elements + block_size - 1) // block_size
    
    # Launch kernel
    fused_arithmetic_kernel[(num_programs,)](
        in_0, in_1, in_2, in_3,
        out,
        n_elements,
        block_size=block_size,
    )
    
    return out

def replacement_func():
    return fused_arithmetic