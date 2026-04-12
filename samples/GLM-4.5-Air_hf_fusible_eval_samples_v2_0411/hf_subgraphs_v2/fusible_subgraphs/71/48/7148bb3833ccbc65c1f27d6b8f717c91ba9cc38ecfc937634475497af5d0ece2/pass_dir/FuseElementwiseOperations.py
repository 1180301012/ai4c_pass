import torch
import triton
import triton.language as tl

@triton.jit
def fused_fma_kernel(
    a_ptr,      # tensor a (tmp_2: ReLU output)
    b_ptr,      # tensor b (expanded param) 
    c_ptr,      # tensor c (residual: tmp_4)
    out_ptr,    # output tensor (tmp_8)
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """Kernel that fuses multiplication and addition operations: a + b * c"""
    pid = tl.program_id(0)
    block_size = BLOCK_SIZE
    start_idx = pid * block_size
    end_idx = min(start_idx + block_size, n_elements)
    
    # Create contiguous offset array
    offsets = start_idx + tl.arange(0, block_size)
    mask = offsets < n_elements
    
    # Load all input tensors
    a = tl.load(a_ptr + offsets, mask=mask, other=0.0)
    b = tl.load(b_ptr + offsets, mask=mask, other=0.0)
    c = tl.load(c_ptr + offsets, mask=mask, other=0.0)
    
    # Fused multiply-add operation
    out = a + b * c
    
    # Store result
    tl.store(out_ptr + offsets, out, mask=mask)

@torch.fx.wrap
def fused_multiply_add(a, b, c):
    """
    Fused operation: a + b * c
    Replaces separate multiplication and addition with FMA
    """
    n_elements = a.numel()
    BLOCK_SIZE = 1024
    num_programs = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    out = torch.empty_like(a)
    
    fused_fma_kernel[(num_programs,)](
        a_ptr=a,
        b_ptr=b,
        c_ptr=c,
        out_ptr=out,
        n_elements=n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out

def pattern(a, b, c):
    """Match: Multiplication followed by Addition: a + b * c"""
    tmp_7 = b * c
    tmp_8 = a + tmp_7
    return tmp_8

def replacement_args(a, b, c):
    """Extract arguments for replacement function"""
    return (a, b, c)

def replacement_func():
    """Return the fused multiply-add function"""
    return fused_multiply_add