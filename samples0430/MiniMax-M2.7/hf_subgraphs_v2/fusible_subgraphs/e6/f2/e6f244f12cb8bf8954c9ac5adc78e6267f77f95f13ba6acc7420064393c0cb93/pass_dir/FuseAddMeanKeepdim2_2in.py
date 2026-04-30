import torch
import triton
import triton.language as tl

@triton.jit
def fused_add_kernel_2in(
    x0_ptr, x1_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a contiguous block of BLOCK_SIZE elements
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offs = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements
    
    # Load both inputs
    x0 = tl.load(x0_ptr + offs, mask=mask, other=0.0)
    x1 = tl.load(x1_ptr + offs, mask=mask, other=0.0)
    
    # Compute sum
    out = x0 + x1
    
    # Store output
    tl.store(out_ptr + offs, out, mask=mask)

@torch.fx.wrap
def fused_add_2in(x0, x1):
    """
    Fused kernel for (x0 + x1)
    """
    n_elements = x0.numel()
    
    # Allocate output tensor
    out = torch.empty_like(x0)
    
    # Launch kernel
    BLOCK_SIZE = 1024
    num_programs = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    fused_add_kernel_2in[(num_programs,)](
        x0, x1,
        out,
        n_elements,
        BLOCK_SIZE,
    )
    
    return out


def pattern(in_0, in_1):
    """
    Match pattern: in_1 + in_0
    For graphs like: tmp_0 = 0 + in_1; tmp_0 += in_0
    """
    return in_1 + in_0


def replacement_args(in_0, in_1):
    return (in_0, in_1)


def replacement_func():
    return fused_add_2in