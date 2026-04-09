import torch
import triton
import triton.language as tl

def pattern(x):
    return x.cumsum(-1)

def replacement_args(x):
    return (x,)

@triton.jit
def cumsum_kernel(
    x_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load input
    x = tl.load(x_ptr + offsets, mask=mask, other=0)
    
    # Compute cumulative sum with parallel prefix approach
    out = tl.cumsum(x, dim=0)
    
    # Store result
    tl.store(out_ptr + offsets, out, mask=mask)

@torch.fx.wrap
def optimized_cumsum(x):
    n_elements = x.numel()
    BLOCK_SIZE = 1024
    num_programs = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    out = torch.empty_like(x)
    
    cumsum_kernel[(num_programs,)](
        x,
        out,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out

def replacement_func():
    return optimized_cumsum
    in_1_ptr,
    out_scalar_ptr,
    out_tensor_ptr,
    batch_size,
    seq_len,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    # Compute cumulative sum and masking in a fused manner
    pid = tl.program_id(0)
    batch_offset = pid * seq_len
    
    # Load input tensors
    in_1_vals = tl.load(in_1_ptr + batch_offset)
    
    # Cumulative sum in shared memory for better performance
    cumsum = tl.zeros(seq_len, dtype=tl.int64)
    cumsum[0] = in_1_vals[0]
    for i in range(1, seq_len):
        cumsum[i] = cumsum[i-1] + in_1_vals[i]
    
    # Subtract 1 and apply masking
    masked_vals = cumsum - 1
    
    # Create result tensor (3, seq_len) by broadcasting
    # Each row gets the same values from masked_vals
    for row in range(3):
        row_offset = batch_offset + row * batch_size * seq_len
        for i in range(seq_len):
            tl.store(out_tensor_ptr + row_offset + i, masked_vals[i])
    
    # Compute max operations
    max_over_batch = tl.max(masked_vals)
    max_over_seq = tl.max(masked_vals)
    
    # Final arithmetic
    final_val = max_over_batch - 8  # Equivalent to (max + 1) - 9
    
    # Store final scalar result
    tl.store(out_scalar_ptr + pid, final_val)

@torch.fx.wrap
def fused_computation_wrapper(in_0, in_1):
    batch_size, seq_len = in_1.shape
    
    # Create output tensors
    out_tensor = torch.empty((3, batch_size, seq_len), dtype=torch.int64, device='cuda')
    out_scalar = torch.empty(batch_size, dtype=torch.int64, device='cuda')
    
    # Handle the computation in chunks if batch_size is large
    if batch_size * seq_len > 65536:  # Use smaller block for large inputs
        BLOCK_SIZE_M = 128
        BLOCK_SIZE_N = 128
    else:
        BLOCK_SIZE_M = 256
        BLOCK_SIZE_N = 256
    
    # Configure grid and execute kernel
    grid = (batch_size,)
    
    fused_kernel[grid](
        in_0,
        in_1,
        out_scalar,
        out_tensor,
        batch_size,
        seq_len,
        BLOCK_SIZE_M,
        BLOCK_SIZE_N,
    )
    
    return out_scalar, out_tensor

def replacement_func():
    return fused_computation_wrapper