import torch
import triton
import triton.language as tl


def pattern(x):
    return torch.nn.functional.hardtanh(x, 0.0, 6.0, False)


def replacement_args(x):
    return x,


@triton.jit
def hardtanh_kernel(x_ptr, out_ptr, n_elements, min_val, max_val, BLOCK_SIZE: tl.constexpr):
    """Hardtanh kernel: out = min(max(x, min_val), max_val)"""
    # Each program handles a contiguous block of data
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load input data
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    
    # Apply hardtanh: clamp between min_val and max_val
    out = tl.maximum(tl.minimum(x, max_val), min_val)
    
    # Store result
    tl.store(out_ptr + offsets, out, mask=mask)


@torch.fx.wrap
def optimized_hardtanh(x, min_val=0.0, max_val=6.0, inplace=False):
    """Optimized Hardtanh implementation using Triton"""
    if inplace:
        raise NotImplementedError("Inplace hardtanh not supported")
    
    # Create output tensor
    out = torch.empty_like(x)
    
    # Calculate total number of elements
    n_elements = x.numel()
    
    # Choose block size - power of 2 for memory coalescing
    BLOCK_SIZE = 1024
    num_programs = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Launch kernel with autotuning
    hardtanh_kernel[(num_programs,)](
        x_ptr=x,
        out_ptr=out,
        n_elements=n_elements,
        min_val=min_val,
        max_val=max_val,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out


def replacement_func():
    return optimized_hardtanh