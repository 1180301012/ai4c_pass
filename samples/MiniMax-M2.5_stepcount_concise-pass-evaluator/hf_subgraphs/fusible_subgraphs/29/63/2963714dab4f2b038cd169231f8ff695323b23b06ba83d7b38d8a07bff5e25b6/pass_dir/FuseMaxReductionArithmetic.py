import torch
import triton
import triton.language as tl


@triton.jit
def fused_max_reduction_kernel(
    in_ptr,
    out_ptr,
    B: tl.constexpr,
    S: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fused kernel that computes:
    1. max over dim=0 (keeping result for each batch/seq position)
    2. max over dim=-1 (with keepdim=True)
    3. + 1
    4. - 9
    
    Input shape: [3, B, S]
    Output shape: [B, 1]
    """
    # We need to compute:
    # tmp_8 = tmp_7.max(0, keepdim=False) -> shape [B, S]
    # tmp_10 = tmp_9.max(-1, keepdim=True) -> shape [B, 1]
    # tmp_12 = tmp_11 + 1
    # tmp_13 = tmp_12 - 9
    
    # Each program computes one batch element
    pid = tl.program_id(0)
    batch_offset = pid * S
    
    # Step 1: Compute max over dim=0 (3 elements)
    # Each program loads all 3 values for each sequence position and takes max
    max_val = tl.zeros((BLOCK_SIZE,), tl.dtype(tl.int64))
    
    for i in range(3):
        offsets = tl.arange(0, BLOCK_SIZE)
        mask = offsets < S
        
        # Load from [i, pid, :]
        in_offset = i * B * S + batch_offset + offsets
        val = tl.load(in_ptr + in_offset, mask=mask, other=0)
        max_val = tl.maximum(max_val, val)
    
    # Now max_val has shape [S], max over dim=0
    
    # Step 2: Compute max over dim=-1 (with keepdim=True, but we just need scalar)
    # We need to find the maximum across all sequence positions
    # For each block, compute local max
    local_max = tl.zeros((BLOCK_SIZE,), tl.dtype(tl.int64))
    for j in range(0, S, BLOCK_SIZE):
        offsets = j + tl.arange(0, BLOCK_SIZE)
        mask = offsets < S
        val = tl.load(in_ptr + batch_offset + offsets, mask=mask, other=0)
        local_max = tl.where(mask, local_max, val)
    
    # Actually we already have max_val from step 1, let's compute the max across S
    # We need to reduce max_val across all S elements
    # This is a tree-reduction
    block_max = max_val
    BLOCK_SIZE_RED = 128
    
    # Warp-level reduction would be better, but let's do a simple approach
    # For now, just compute max directly by loading all values
    all_max = tl.zeros((1,), tl.dtype(tl.int64))
    for j in range(0, S, BLOCK_SIZE_RED):
        offsets = j + tl.arange(0, BLOCK_SIZE_RED)
        mask = offsets < S
        val = tl.load(in_ptr + batch_offset + offsets, mask=mask, other=0)
        all_max = tl.maximum(all_max, val)
    
    # Step 3 & 4: +1 and -9
    result = all_max + 1 - 9
    
    # Store result at [pid, 0]
    tl.store(out_ptr + pid, result)


@torch.fx.wrap
def fused_max_reduction_arithmetic(in_tensor):
    """
    Fused kernel for max reduction + arithmetic operations.
    
    Replaces:
    - tmp_8 = tmp_7.max(0, keepdim=False)
    - tmp_10 = tmp_9.max(-1, keepdim=True)
    - tmp_12 = tmp_11 + 1
    - tmp_13 = tmp_12 - 9
    
    Input: [3, B, S] tensor on cuda
    Output: [B, 1] tensor
    """
    B = in_tensor.shape[1]
    
    # Allocate output
    out = torch.empty((B, 1), dtype=torch.int64, device='cuda')
    
    # Configure kernel
    BLOCK_SIZE = 128
    num_programs = B
    
    fused_max_reduction_kernel[(num_programs)](
        in_ptr=in_tensor,
        out_ptr=out,
        B=B,
        S=in_tensor.shape[2],
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out


def pattern(in_tensor):
    """Match the max reduction + arithmetic pattern"""
    tmp_8 = in_tensor.max(0, keepdim=False)
    tmp_9 = tmp_8[0]
    tmp_10 = tmp_9.max(-1, keepdim=True)
    tmp_11 = tmp_10[0]
    tmp_12 = tmp_11 + 1
    tmp_13 = tmp_12 - 9
    return tmp_13


def replacement_args(in_tensor):
    return (in_tensor,)


def replacement_func():
    return fused_max_reduction_arithmetic