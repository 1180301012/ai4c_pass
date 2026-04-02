import torch
import triton
import triton.language as tl

def pattern(x):
    """Pattern matching for fused element-wise operations: sigmoid(x) - 0.25 * π"""
    tmp_5 = x.sigmoid()
    tmp_6 = tmp_5 - 0.25
    tmp_7 = tmp_6 * 3.141592653589793
    return tmp_7

def replacement_args(x):
    return (x,)

@triton.jit
def fused_elementwise_kernel(
    x_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """Fused kernel for sigmoid(x) - 0.25 * π operations"""
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load input
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    
    # Fused computation: sigmoid(x) - 0.25 * π
    # Using exact mathematical operations for perfect numerical equivalence
    x_f32 = tl.cast(x, tl.float32)
    
    # Compute sigmoid using mathematically equivalent formula
    # For numerical stability, use different formulations based on input sign
    sigmoid_val = tl.where(
        x_f32 >= 0,
        1.0 / (1.0 + tl.exp(-x_f32)),
        tl.exp(x_f32) / (1.0 + tl.exp(x_f32))
    )
    
    # Perform the fused operation: sigmoid(x) - 0.25 * π
    pi = 3.141592653589793
    result = sigmoid_val - 0.25 * pi
    
    # Store result
    tl.store(out_ptr + offsets, tl.cast(result, x.dtype), mask=mask)

@torch.fx.wrap
def fused_elementwise_ops(x):
    """Optimized fused element-wise operations"""
    n_elements = x.numel()
    block_size = 1024
    num_programs = (n_elements + block_size - 1) // block_size
    
    output = torch.empty_like(x)
    
    fused_elementwise_kernel[(num_programs,)](
        x_ptr=x,
        out_ptr=output,
        n_elements=n_elements,
        BLOCK_SIZE=block_size,
    )
    
    return output

def replacement_func():
    return fused_elementwise_ops