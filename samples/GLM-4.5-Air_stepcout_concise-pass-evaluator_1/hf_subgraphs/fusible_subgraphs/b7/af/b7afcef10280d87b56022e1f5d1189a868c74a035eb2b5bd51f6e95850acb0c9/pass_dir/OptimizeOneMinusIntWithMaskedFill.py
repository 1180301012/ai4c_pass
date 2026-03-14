import torch
import triton
import triton.language as tl

def pattern(in_0):
    tmp_0 = in_0.to(torch.float32)
    return (tmp_0,)

def replacement_args(in_0):
    return (in_0,)

@triton.jit
def optimized_kernel(
    in_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    # Initialize program coordinates
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load integer input directly (values are 0 or 1)
    x_int = tl.load(in_ptr + offsets, mask=mask, other=0)
    
    # Optimized computation: Perform just the conversion (not the entire chain)
    # Since the pattern matches only the conversion, we should return the float32 version
    result = x_int.to(tl.float32)
    
    # Store result
    tl.store(out_ptr + offsets, result, mask=mask)

@torch.fx.wrap
def kernel_wrapper(in_0):
    N = in_0.numel()
    BLOCK_SIZE = 1024  # Optimized for GPU occupancy
    num_programs = (N + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    out = torch.empty(N, dtype=torch.float32, device=in_0.device)
    
    optimized_kernel[(num_programs,)](
        in_ptr=in_0,
        out_ptr=out,
        n_elements=N,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    # Reshape output to match input shape
    out = out.view(in_0.shape)
    return out

def replacement_func():
    return kernel_wrapper