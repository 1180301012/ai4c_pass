import torch
import triton
import triton.language as tl

def pattern(b, c, a, d):
    # Simple element-wise operations pattern
    x = b * c
    y = a * d
    z = x + y
    return z

def replacement_args(b, c, a, d):
    return (b, c, a, d)

@triton.jit
def simple_elementwise_kernel(
    b_ptr, c_ptr, a_ptr, d_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr
):
    # Each program handles a contiguous block of data
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load all inputs with vectorized loads for better performance
    b = tl.load(b_ptr + offsets, mask=mask, other=0.0)
    c = tl.load(c_ptr + offsets, mask=mask, other=0.0)
    a = tl.load(a_ptr + offsets, mask=mask, other=0.0)
    d = tl.load(d_ptr + offsets, mask=mask, other=0.0)
    
    # Fused computation without intermediate variables
    # This reduces register pressure and memory operations
    out = (b * c) + (a * d)
    
    # Store result
    tl.store(out_ptr + offsets, out, mask=mask)

@torch.fx.wrap
def simple_elementwise_fusion(b, c, a, d):
    # Get tensor size
    n_elements = b.numel()
    
    # Use larger block size for better GPU occupancy on this workload
    # For ~76,800 elements, block sizes of 256 or 512 might be optimal
    BLOCK_SIZE = 512
    num_programs = triton.cdiv(n_elements, BLOCK_SIZE)
    
    # Allocate output tensor with cache-friendly memory
    out = torch.empty_like(b)
    
    # Launch kernel
    simple_elementwise_kernel[(num_programs,)](
        b_ptr=b,
        c_ptr=c,
        a_ptr=a,
        d_ptr=d,
        out_ptr=out,
        n_elements=n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return out

def replacement_func():
    return simple_elementwise_fusion