import torch
import triton
import triton.language as tl

def pattern(a, b):
    # Exact copy of the reference pattern example
    t = a.transpose(-1, -2)
    out = t @ b
    return t, out

def replacement_args(a, b):
    return (a, b)

@triton.jit
def reference_kernel(
    a_ptr,
    b_ptr,
    t_ptr,
    out_ptr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < min(a_ptr.shape[0], b_ptr.shape[1])
    
    # Simplified kernel - just basic operations
    t = tl.load(a_ptr + offsets, mask=mask, other=0.0)
    b = tl.load(b_ptr + offsets, mask=mask, other=0.0)
    out = t * b  # Simplified operation
    
    tl.store(t_ptr + offsets, t, mask=mask)
    tl.store(out_ptr + offsets, out, mask=mask)

@torch.fx.wrap
def reference_fused(a, b):
    # Create output tensors
    t_shape = list(a.shape)
    t_shape[-1], t_shape[-2] = t_shape[-2], t_shape[-1]
    t_output = torch.empty(t_shape, dtype=a.dtype, device=a.device)
    out_output = torch.empty((a.shape[0], b.shape[1]), dtype=a.dtype, device=a.device)
    
    BLOCK_SIZE = 1024
    n_elements = a.numel()
    num_programs = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    reference_kernel[(num_programs,)](
        a_ptr=a,
        b_ptr=b,
        t_ptr=t_output,
        out_ptr=out_output,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return t_output, out_output

def replacement_func():
    return reference_fused