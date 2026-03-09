import torch
import triton
import triton.language as tl

def pattern(x):
    # Match the pattern: convert float32 to float16
    x_float16 = x.to(dtype=torch.float16)
    return x_float16

def replacement_args(x):
    return (x,)

@triton.jit
def triton_type_conversion_kernel(
    in_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a contiguous block of data
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load input data as float32
    x = tl.load(in_ptr + offsets, mask=mask, other=0.0)
    
    # Convert to float16 and store
    tl.store(out_ptr + offsets, x.to(tl.float16), mask=mask)

@torch.fx.wrap
def triton_type_conversion(x):
    # Get input tensor info
    N = x.numel()
    
    # Optimal block size for GPU
    BLOCK_SIZE = 1024
    num_programs = (N + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Create output tensor with float16 dtype
    out = torch.empty(x.shape, dtype=torch.float16, device=x.device)
    
    # Launch the kernel
    triton_type_conversion_kernel[(num_programs,)](
        in_ptr=x,
        out_ptr=out,
        n_elements=N,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out

def replacement_func():
    return triton_type_conversion