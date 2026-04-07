import torch
import triton
import triton.language as tl

def pattern(in_9, tmp_9):
    """
    Pattern matching for the double sigmoid operations:
    tmp_10 = in_9.sigmoid()
    tmp_11 = tmp_9.sigmoid()
    """
    tmp_10 = in_9.sigmoid()
    tmp_11 = tmp_9.sigmoid()
    return tmp_10, tmp_11

def replacement_args(in_9, tmp_9):
    return (in_9, tmp_9)

@triton.jit
def fused_double_sigmoid_kernel(
    a_ptr,
    b_ptr, 
    out1_ptr,
    out2_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """Fused kernel for computing sigmoid(a) and sigmoid(b)"""
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # Create mask to handle boundary conditions
    mask = offsets < n_elements
    
    # Load input tensors
    a = tl.load(a_ptr + offsets, mask=mask, other=0.0)
    b = tl.load(b_ptr + offsets, mask=mask, other=0.0)
    
    # Compute fused sigmoid operations
    # Using fast sigmoid approximation: 1 / (1 + exp(-x)) ≈ 1 / (1 + exp(-x))
    # For better numerical stability, we'll use the standard formula
    exp_a = tl.exp(-tl.abs(a))
    exp_b = tl.exp(-tl.abs(b))
    
    mask_pos_a = a >= 0
    mask_pos_b = b >= 0
    
    sigmoid_a = tl.where(mask_pos_a, 1.0 / (1.0 + exp_a), exp_a / (1.0 + exp_a))
    sigmoid_b = tl.where(mask_pos_b, 1.0 / (1.0 + exp_b), exp_b / (1.0 + exp_b))
    
    # Store results
    tl.store(out1_ptr + offsets, sigmoid_a, mask=mask)
    tl.store(out2_ptr + offsets, sigmoid_b, mask=mask)

@torch.fx.wrap
def fused_double_sigmoid(a, b):
    """Wrapper function for fused double sigmoid operations"""
    # Determine the shape based on input tensors
    n_elements = a.numel()
    
    # Set up Triton kernel parameters
    BLOCK_SIZE = 1024
    num_programs = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Create output tensors
    out1 = torch.empty_like(a)
    out2 = torch.empty_like(b)
    
    # Launch Triton kernel
    fused_double_sigmoid_kernel[(num_programs,)](
        a_ptr=a,
        b_ptr=b,
        out1_ptr=out1,
        out2_ptr=out2,
        n_elements=n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out1, out2

def replacement_func():
    return fused_double_sigmoid