import torch
import triton
import triton.language as tl

def pattern(tensor1, tensor2):
    """Pattern to match simple addition operation"""
    result = tensor1 + tensor2
    return result

def replacement_args(tensor1, tensor2):
    return (tensor1, tensor2)

@triton.jit
def optimized_add_kernel(
    x_ptr,
    y_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    """Optimized addition kernel with better tile sizes"""
    pid = tl.program_id(0)
    
    # Calculate work distribution
    grid_m = (n_elements + BLOCK_SIZE_M * BLOCK_SIZE_N - 1) // (BLOCK_SIZE_M * BLOCK_SIZE_N)
    pid_m = pid // grid_m
    pid_n = pid % grid_m
    
    start_n = pid_n * BLOCK_SIZE_N
    start_m = pid_m * BLOCK_SIZE_M
    offsets = start_m + start_n + tl.arange(0, BLOCK_SIZE_M * BLOCK_SIZE_N)
    mask = offsets < n_elements
    
    # Load inputs with better memory coalescing
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = tl.load(y_ptr + offsets, mask=mask, other=0.0)
    
    # Compute addition
    out = x + y
    
    # Store result
    tl.store(out_ptr + offsets, out, mask=mask)

@torch.fx.wrap
def triton_add(x, y):
    N = x.numel()
    
    # Optimized block sizes for better GPU utilization
    BLOCK_SIZE_M = 8
    BLOCK_SIZE_N = 128
    BLOCK_SIZE = BLOCK_SIZE_M * BLOCK_SIZE_N
    
    num_programs = (N + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    out = torch.empty_like(x, dtype=torch.float16)
    
    optimized_add_kernel[(num_programs,)](
        x_ptr=x,
        y_ptr=y,
        out_ptr=out,
        n_elements=N,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
    )
    
    return out

def replacement_func():
    return triton_add