import torch
import triton
import triton.language as tl

# Pattern matching function
def pattern(in_2, in_1, in_4):
    """
    Pattern: Key states operations with splitting, negation, and element-wise operations
    
    Matches:
    tmp_0 = in_2 * in_1
    tmp_1 = in_2[Ellipsis, slice(None, 128, None)]
    tmp_2 = in_2[Ellipsis, slice(128, None, None)]
    tmp_3 = -tmp_2
    tmp_4 = torch.cat((tmp_3, tmp_1), dim=-1)
    tmp_5 = tmp_4 * in_4
    tmp_6 = tmp_0 + tmp_5
    
    The pattern: For each element in in_2:
    - First half (0-127): result = (in_2 * in_1) + (-in_2 * in_4)
    - Second half (128-255): result = (in_2 * in_1) + (in_2 * in_4)  
    """
    tmp_0 = in_2 * in_1
    tmp_1 = in_2[Ellipsis, slice(None, 128, None)]
    tmp_2 = in_2[Ellipsis, slice(128, None, None)]
    tmp_3 = -tmp_2
    tmp_4 = torch.cat((tmp_3, tmp_1), dim=-1)
    tmp_5 = tmp_4 * in_4
    tmp_6 = tmp_0 + tmp_5
    # Need to return all observable intermediates from original computation
    return tmp_6

# Argument extraction function
def replacement_args(in_2, in_1, in_4):
    return (in_2, in_1, in_4)

# Optimized kernel with performance tuning
@triton.jit
def optimized_key_operations_kernel(
    key_states_ptr,
    cos_ptr,
    sin_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Optimized kernel with better performance configuration
    """
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load inputs with optimized memory access
    key_states = tl.load(key_states_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    cos = tl.load(cos_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    sin = tl.load(sin_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    
    # Optimized computation: fused operation to minimize memory access
    # Original pattern: tmp_0 = key_states * cos, tmp_5 involves negation combination
    final_result = (key_states * cos) - (key_states * sin)
    
    # Convert back to bfloat16 and store
    tl.store(out_ptr + offsets, final_result.to(tl.bfloat16), mask=mask)

@torch.fx.wrap
def fused_key_states_operations(in_2, in_1, in_4):
    """
    Wrapper function for the optimized kernel
    """
    total_elements = in_2.numel()
    
    # Optimal block size for better GPU utilization
    BLOCK_SIZE = 256
    
    # Calculate number of programs
    num_programs = (total_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Create output tensor
    out = torch.empty_like(in_2)
    
    # Launch optimized kernel
    optimized_key_operations_kernel[(num_programs,)](
        key_states_ptr=in_2,
        cos_ptr=in_1,
        sin_ptr=in_4,
        out_ptr=out,
        n_elements=total_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out

# Replacement function (NO arguments, returns function reference)
def replacement_func():
    return fused_key_states_operations