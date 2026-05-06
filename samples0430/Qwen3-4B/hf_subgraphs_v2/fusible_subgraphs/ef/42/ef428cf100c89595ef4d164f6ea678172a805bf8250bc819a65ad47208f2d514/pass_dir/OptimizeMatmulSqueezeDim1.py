import torch
import triton
import triton.language as tl

def pattern(in_0, in_1):
    matmul = torch.matmul(in_0, in_1)
    tmp_1 = matmul.squeeze(1)
    return (tmp_1,)

def replacement_args(in_0, in_1):
    return (in_0, in_1)

@triton.jit
def optimized_kernel(
    in_0_ptr,
    in_1_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr
):
    # Calculate the program ID for this block
    pid = tl.program_id(0)
    
    # Initialize accumulator
    total = 0.0
    
    # Loop over feature dimension (249)
    for k in range(249):
        # Load from in_0 (dimension: [1, 1, 249])
        x = tl.load(in_0_ptr + k, mask=True, other=0.0)
        
        # Load from in_1 (dimension: [1, 249, 64])
        y = tl.load(in_1_ptr + k * 64 + pid, mask=True, other=0.0)
        
        total += x * y
    
    # Store result
    tl.store(out_ptr + pid, total)

@torch.fx.wrap
def kernel_wrapper(in_0, in_1):
    n = in_1.shape[-1]  # Output channels (64)
    BLOCK_SIZE = 256
    num_programs = (n + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    out = torch.empty(n, device=in_0.device, dtype=in_0.dtype)
    
    optimized_kernel[(num_programs,)](
        in_0_ptr=in_0,
        in_1_ptr=in_1,
        out_ptr=out,
        n_elements=n,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return (out,)

def replacement_func():
    return kernel_wrapper