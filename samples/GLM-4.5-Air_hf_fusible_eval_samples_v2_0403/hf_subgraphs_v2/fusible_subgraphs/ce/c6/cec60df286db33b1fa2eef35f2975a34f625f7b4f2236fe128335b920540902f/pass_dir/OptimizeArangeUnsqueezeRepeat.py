import torch
import triton
import triton.language as tl

# Pattern matching function
def pattern():
    """Match the pattern: arange → unsqueeze → repeat(1, 1)"""
    tmp_0 = torch.arange(0, 1, device=device(type='cuda', index=0))
    tmp_1 = tmp_0.unsqueeze(0)
    tmp_2 = tmp_1.repeat(1, 1)
    return tmp_0, tmp_2

# Argument extraction function
def replacement_args():
    return ()

# Optimized kernel that directly creates both tensors at their final shapes
@triton.jit
def optimized_arange_kernel(output_1_ptr, output_2_ptr, n_elements_2d, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0) * BLOCK_SIZE
    offsets = pid + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements_2d
    
    # Compute 0. value for both tensors
    val = tl.full((BLOCK_SIZE,), 0.0, dtype=tl.float16)
    
    # Store to output tensors with proper broadcasting
    # tmp_0: shape [1] - we'll create the 1D tensor
    # tmp_2: shape [1, 1] - we'll create the 2D tensor
    
    # For the 2D output, we need to handle the broadcasting properly
    # Since we only have one element, we just store it in the first position
    if mask[0]:
        tl.store(output_1_ptr, val[0])
        tl.store(output_2_ptr, val[0])

@torch.fx.wrap
def optimized_arange_operations():
    # Create range for tmp_0 directly
    tmp_0 = torch.arange(0, 1, dtype=torch.float16, device='cuda')
    
    # Create expanded 2D tensor directly for tmp_2
    tmp_2 = torch.full((1, 1), 0.0, dtype=torch.float16, device='cuda')
    
    return tmp_0, tmp_2

# Replacement function (returns the optimized wrapper)
def replacement_func():
    return optimized_arange_operations