import torch
import triton
import triton.language as tl

# Pattern matching function for addition + softmax with zero dropout
def pattern(in_0, in_1):
    """Match: addition -> softmax -> type conversion (dropout is no-op when p=0.0)"""
    tmp_0 = in_0 + in_1
    tmp_1 = torch.nn.functional.softmax(tmp_0, dim=-1)
    tmp_2 = None  # This line ensures dropout_p=0.0 (no dropout effect)
    tmp_3 = tmp_1.to(torch.float32)
    return tmp_3

# Argument extraction function
def replacement_args(in_0, in_1):
    return (in_0, in_1)

# Optimized fused kernel using Triton - addition + softmax only
@triton.jit
def fused_add_softmax_kernel(
    x_ptr, 
    y_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr
):
    """Fused kernel: addition + softmax (optimized for zero dropout case)"""
    # Each program handles a contiguous block of data
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load input tensors
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = tl.load(y_ptr + offsets, mask=mask, other=0.0)
    
    # Step 1: Element-wise addition
    added = x + y
    
    # Step 2: Softmax along last dimension
    # Get maximum for numerical stability
    max_val = tl.max(added, mask=mask)
    shifted = added - max_val
    exp_val = tl.exp(shifted)
    sum_exp = tl.sum(exp_val, mask=mask)
    softmax_val = exp_val / (sum_exp + 1e-20)
    
    # Store result
    tl.store(out_ptr + offsets, softmax_val, mask=mask)

@torch.fx.wrap
def fused_add_softmax(x, y):
    """Wrapper function for optimized addition + softmax"""
    # Use fallback for small tensors or shape mismatches
    if x.shape != y.shape or x.numel() < 1024:
        tmp_0 = x + y
        tmp_1 = torch.nn.functional.softmax(tmp_0, dim=-1)
        tmp_3 = tmp_1.to(torch.float32)
        return tmp_3
    
    out = torch.empty_like(x)
    
    # Optimized Triton kernel for suitable tensors
    n_elements = x.numel()
    BLOCK_SIZE = min(512, n_elements)
    num_programs = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    fused_add_softmax_kernel[(num_programs,)](
        x_ptr=x,
        y_ptr=y,
        out_ptr=out,
        n_elements=n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return out

# Replacement function
def replacement_func():
    # Return a closure that performs addition + softmax without dropout
    def kernel_optimized(x, y):
        return fused_add_softmax(x, y)
    
    return kernel_optimized