import torch
import triton
import triton.language as tl

def pattern(x):
    # Match the unsqueeze + subtraction pattern
    # x is the input tensor [1, 361, 49]
    tmp_10 = x.unsqueeze(2)
    tmp_11 = x.unsqueeze(3)
    tmp_12 = tmp_10 - tmp_11
    return tmp_12

def replacement_args(x):
    return (x,)

@triton.jit
def optimized_subtract_kernel(
    in_ptr,
    out_ptr,
    n_batch, n_seq, n_dim1,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < (n_batch * n_seq * n_dim1 * n_dim1)
    
    # Compute indices
    offset = offsets
    batch = offset // (n_seq * n_dim1 * n_dim1)
    remaining = offset % (n_seq * n_dim1 * n_dim1)
    seq = remaining // (n_dim1 * n_dim1)
    remaining = remaining % (n_dim1 * n_dim1)
    dim1_j = remaining // n_dim1
    dim1_i = remaining % n_dim1
    
    # Compute memory access pattern
    # Original tensor: [batch, seq, dim1]
    # tmp_10: [batch, seq, 1, dim1] -> flatten: [batch, seq, 1, dim1]
    # tmp_11: [batch, seq, dim1, 1] -> flatten: [batch, seq, dim1, 1]
    # Output: [batch, seq, dim1, dim1]
    
    # Index in original tensor for tmp_10 (unsqueeze at dim=2)
    idx1 = batch * (n_seq * n_dim1) + seq * n_dim1 + dim1_j
    
    # Index in original tensor for tmp_11 (unsqueeze at dim=3)
    idx2 = batch * (n_seq * n_dim1) + seq * n_dim1 + dim1_i
    
    # Load both elements
    val1 = tl.load(in_ptr + idx1, mask=mask)
    val2 = tl.load(in_ptr + idx2, mask=mask)
    
    # Subtract
    result = val1 - val2
    
    # Store result
    out_idx = batch * (n_seq * n_dim1 * n_dim1) + seq * (n_dim1 * n_dim1) + dim1_j * n_dim1 + dim1_i
    tl.store(out_ptr + out_idx, result, mask=mask)

@torch.fx.wrap
def optimized_unsqueeze_subtract(tmp_9):
    batch, seq, dim1 = tmp_9.shape
    total_elements = batch * seq * dim1 * dim1
    BLOCK_SIZE = 1024
    num_programs = (total_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    out = torch.empty((batch, seq, dim1, dim1), dtype=tmp_9.dtype, device=tmp_9.device)
    
    optimized_subtract_kernel[(num_programs,)](
        tmp_9,
        out,
        batch, seq, dim1,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out

def replacement_func():
    return optimized_unsqueeze_subtract