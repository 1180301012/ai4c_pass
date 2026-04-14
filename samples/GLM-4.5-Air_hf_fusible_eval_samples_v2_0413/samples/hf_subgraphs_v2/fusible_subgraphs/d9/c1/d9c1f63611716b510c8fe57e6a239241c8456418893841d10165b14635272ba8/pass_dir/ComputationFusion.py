import torch
import triton
import triton.language as tl

def pattern(x1, x2, x3, x4):
    # Pattern: Multiple independent element-wise operations that can be fused
    # This represents operations like sigmoid * layer_norm unsqueeze + sigmoid * layer_norm
    
    # This is a simplified pattern that captures the essence of:
    # tmp_15 = tmp_11 * tmp_14
    # tmp_16 = tmp_10 * tmp_13  
    # tmp_17 = tmp_15 + tmp_16
    
    # But we'll do it as: (x1 * x2) + (x3 * x4)
    result = (x1 * x2) + (x3 * x4)
    return result

def replacement_args(x1, x2, x3, x4):
    return (x1, x2, x3, x4)

@triton.jit
def fused_computation_kernel(
    x1_ptr, x2_ptr, x3_ptr, x4_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr
):
    # Each program handles a contiguous block of data
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load all inputs with optimized memory access
    x1 = tl.load(x1_ptr + offsets, mask=mask, other=0.0)
    x2 = tl.load(x2_ptr + offsets, mask=mask, other=0.0)
    x3 = tl.load(x3_ptr + offsets, mask=mask, other=0.0)
    x4 = tl.load(x4_ptr + offsets, mask=mask, other=0.0)
    
    # Fused computation with vectorized operations
    # Combine multiplication and addition in one step to reduce operations
    result = (x1 * x2) + (x3 * x4)
    
    # Store result
    tl.store(out_ptr + offsets, result, mask=mask)

@torch.fx.wrap
def fused_computation(x1, x2, x3, x4):
    # Get tensor size
    n_elements = x1.numel()
    
    # Optimize block size for the workload (~76,800 elements)
    # For this size, we want fewer, larger blocks to reduce overhead
    BLOCK_SIZE = 2048  # Larger block size to reduce kernel launch overhead
    num_programs = triton.cdiv(n_elements, BLOCK_SIZE)
    
    # Allocate output tensor
    out = torch.empty_like(x1)
    
    # Launch kernel with optimized grid size
    fused_computation_kernel[(num_programs,)](
        x1_ptr=x1,
        x2_ptr=x2,
        x3_ptr=x3,
        x4_ptr=x4,
        out_ptr=out,
        n_elements=n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return out

def replacement_func():
    return fused_computation