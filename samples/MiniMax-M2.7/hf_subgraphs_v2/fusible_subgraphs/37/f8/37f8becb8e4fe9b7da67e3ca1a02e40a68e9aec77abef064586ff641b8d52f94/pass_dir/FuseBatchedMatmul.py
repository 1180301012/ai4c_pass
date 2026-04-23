import torch
from torch import device
import triton
import triton.language as tl

# Pattern matching function - match ONLY the matmul operation
def pattern(a, b):
    """
    Match pattern: matmul(a, b) where a is [2, D] and b is [D, 1]
    """
    return torch.matmul(a, b)

# Argument extraction function
def replacement_args(a, b):
    return (a, b)

# Triton kernel for optimized batched matmul [2, D] @ [D, 1]
@triton.jit
def batched_matmul_kernel(
    in_2_ptr, in_3_ptr, out_ptr,
    D: tl.constexpr, BLOCK_SIZE: tl.constexpr,
    dtype: tl.constexpr
):
    """
    Batched matmul kernel for [2, D] @ [D, 1] -> [2, 1]
    """
    # Each program handles one output element (row 0 or 1)
    batch_idx = tl.program_id(0)
    
    # Compute starting pointers for this batch
    row_offset = batch_idx * D
    in_2_row_ptr = in_2_ptr + row_offset
    
    # Initialize accumulator as scalar (float32 for precision)
    acc = 0.0
    
    # Blocked computation for better cache utilization
    for start in range(0, D, BLOCK_SIZE):
        # Compute offsets for this block
        offsets = start + tl.arange(0, BLOCK_SIZE)
        # Mask to ensure we don't go out of bounds
        mask = offsets < D
        
        # Load from in_2 (row vector) - cast to float32 for accumulation
        a = tl.load(in_2_row_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        
        # Load from in_3 (column vector) - cast to float32 for accumulation
        b = tl.load(in_3_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        
        # Compute dot product for this block
        acc += tl.sum(a * b)
    
    # Store result - cast back to original dtype
    out_idx = batch_idx
    result = acc.to(dtype)
    tl.store(out_ptr + out_idx, result)

# Helper to get Triton dtype from torch dtype
def get_triton_dtype(torch_dtype):
    if torch_dtype == torch.float16:
        return tl.float16
    elif torch_dtype == torch.bfloat16:
        return tl.bfloat16
    elif torch_dtype == torch.float32:
        return tl.float32
    else:
        return tl.float32

# Optimized wrapper function with @torch.fx.wrap decorator
@torch.fx.wrap
def optimized_batched_matmul(a, b):
    """
    Optimized batched matmul
    """
    M, D = a.shape  # M=2, D varies (768 or 1152)
    
    # Allocate output tensor
    out = torch.empty((M, 1), dtype=a.dtype, device=a.device)
    
    # Choose block size based on dimension
    if D <= 768:
        BLOCK_SIZE = 128
    elif D <= 1152:
        BLOCK_SIZE = 256
    else:
        BLOCK_SIZE = 512
    
    # Get Triton dtype
    triton_dtype = get_triton_dtype(a.dtype)
    
    # Launch kernel with 2 programs (one per output row)
    grid = (M,)
    batched_matmul_kernel[grid](
        a, b, out,
        D, BLOCK_SIZE, triton_dtype
    )
    
    return out

# Replacement function (NO arguments, returns function reference)
def replacement_func():
    return optimized_batched_matmul