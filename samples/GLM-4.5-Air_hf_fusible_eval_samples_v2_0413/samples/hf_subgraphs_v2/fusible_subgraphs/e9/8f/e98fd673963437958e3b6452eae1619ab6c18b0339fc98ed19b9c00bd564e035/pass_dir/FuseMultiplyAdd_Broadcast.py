import torch
import triton
import triton.language as tl

@triton.jit
def fused_multiply_add_kernel(
    in_0_ptr,
    in_1_ptr,
    sigmoid_expanded_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load inputs
    in_0_val = tl.load(in_0_ptr + offsets, mask=mask)
    in_1_val = tl.load(in_1_ptr + offsets, mask=mask)
    sigmoid_val = tl.load(sigmoid_expanded_ptr + offsets, mask=mask)
    
    # Fused multiply-add: result = (in_1 * sigmoid_expanded) + in_0
    multiply_val = in_1_val * sigmoid_val
    result = multiply_val + in_0_val
    
    # Store result
    tl.store(out_ptr + offsets, result, mask=mask)

@torch.fx.wrap
def fused_multiply_add(in_0, in_1, sigmoid_expanded):
    """Fuses multiply + add into single operation"""
    # Use Triton for optimal performance
    N = in_0.numel()
    BLOCK_SIZE = 1024
    num_programs = (N + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Create output tensor
    out = torch.empty_like(in_0)
    
    # Launch Triton kernel
    fused_multiply_add_kernel[(num_programs,)](
        in_0_ptr=in_0,
        in_1_ptr=in_1,
        sigmoid_expanded_ptr=sigmoid_expanded,
        out_ptr=out,
        n_elements=N,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out

def pattern(tmp_3, tmp_4, in_0, in_1, sigmoid_expanded):
    # Match: tmp_3 = in_1 * sigmoid_expanded + in_0
    # tmp_3 = in_1 * sigmoid_expanded
    multiply_result = in_1 * sigmoid_expanded
    # tmp_3 += in_0
    add_result = multiply_result + in_0
    
    return add_result

def replacement_args(tmp_3, tmp_4, in_0, in_1, sigmoid_expanded):
    return (in_0, in_1, sigmoid_expanded)

@torch.fx.wrap  
def fused_multiply_add(in_0, in_1, sigmoid_expanded):
    """Fuses multiply + add into single operation"""
    # Use Triton for optimal performance
    N = in_0.numel()
    BLOCK_SIZE = 1024
    num_programs = (N + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Create output tensor
    out = torch.empty_like(in_0)
    
    # Launch Triton kernel
    fused_multiply_add_kernel[(num_programs,)](
        in_0_ptr=in_0,
        in_1_ptr=in_1,
        sigmoid_expanded_ptr=sigmoid_expanded,
        out_ptr=out,
        n_elements=N,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out

def replacement_func():
    return fused_multiply_add