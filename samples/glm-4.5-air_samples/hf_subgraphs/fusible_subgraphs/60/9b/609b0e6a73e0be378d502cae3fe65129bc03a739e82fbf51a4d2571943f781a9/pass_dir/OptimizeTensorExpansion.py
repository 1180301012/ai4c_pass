import torch
import triton
import triton.language as tl

def pattern(gamma, unsqueezed_part):
    # gamma[None, None, slice(None, None, None)] creates expansion
    gamma_expanded = gamma[None, None, slice(None, None, None)]
    # This pattern combines the expansion with the unsqueezed tensor
    return gamma_expanded, unsqueezed_part

def replacement_args(gamma, unsqueezed_part):
    return (gamma, unsqueezed_part)

@triton.jit
def optimized_expansion_kernel(
    gamma_ptr,
    unsqueezed_part_ptr,
    output_ptr,
    gamma_size,
    unsqueezed_part_size,
    BLOCK_SIZE: tl.constexpr,
):
    """Optimized kernel for tensor expansion operations"""
    
    # Handle gamma expansion: [2] -> [1, 1, 2]
    gamma_pid = tl.program_id(0)
    if gamma_pid == 0:
        for i in range(gamma_size):
            tl.store(output_ptr + i, tl.load(gamma_ptr + i))
    
    # Handle unsqueezed part (already in correct shape)
    # This kernel mainly ensures memory coalescing for the expansion

@torch.fx.wrap  
def optimized_tensor_expansion(gamma, unsqueezed_part):
    """Optimized tensor expansion that avoids unnecessary operations"""
    
    # gamma: [2, 128] -> [1, 1, 2, 128] or whatever the target shape is
    # Let me check what the original operation actually does:
    # tmp_7 = tmp_0[None, None, slice(None, None, None)]
    # tmp_0 has shape [2, 128], so this creates [1, 1, 2, 128]
    
    gamma_expanded = gamma.view(1, 1, *gamma.shape)
    
    # The unsqueezed_part is already in the correct shape from previous operations
    # Just return both tensors
    
    return gamma_expanded, unsqueezed_part

def replacement_func():
    return optimized_tensor_expansion