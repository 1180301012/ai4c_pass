import torch
import triton
import triton.language as tl

def pattern(in_0, in_1, scalar):
    # Strictly matches the computation pattern without cleanup
    matmul = torch.matmul(in_0, in_1)
    return matmul / scalar
def replacement_args(in_0, in_1, scalar):
    # Extract arguments needed for the replacement kernel
    return (in_0, in_1, scalar)

@triton.jit
def optimized_div_kernel(
    in_0_ptr,
    in_1_ptr,
    scalar,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    # Each thread block handles a contiguous block of elements
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # Load matrices (with proper alignment to minimize memory operations)
    in_0 = tl.load(in_0_ptr + offsets, mask=mask, other=0.0)
    in_1 = tl.load(in_1_ptr + offsets, mask=mask, other=0.0)

    # Compute matrix multiplication (column-major for better GPU memory access)
    out = in_0 @ in_1 / scalar

    # Store results with proper masking
    tl.store(out_ptr + offsets, out, mask=mask)

def kernel_wrapper(in_0, in_1, scalar):
    N = in_0.numel()
    BLOCK_SIZE = 1024
    num_programs = (N + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    out = torch.empty_like(in_0)
    
    optimized_div_kernel[(num_programs,)](\n        in_0_ptr=in_0,
        in_1_ptr=in_1,
        scalar=scalar,
        out_ptr=out,
        n_elements=N,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out
def replacement_func():
    return kernel_wrapper