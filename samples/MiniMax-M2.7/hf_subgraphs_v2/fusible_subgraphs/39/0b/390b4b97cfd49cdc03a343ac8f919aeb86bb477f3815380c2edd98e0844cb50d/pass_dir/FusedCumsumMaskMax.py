import torch
import triton
import triton.language as tl
from torch import device

@triton.jit
def fused_max_arith_kernel(
    tmp7_ptr, result_ptr,
    batch_size, seq_len, heads,
    BLOCK_SIZE: tl.constexpr
):
    # Each program handles one batch element
    pid_batch = tl.program_id(0)
    
    # Process the full sequence for this batch
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < seq_len
    
    # Load values for all 3 heads and find max
    max_val = -9223372036854775807  # min int64
    
    for head_idx in range(heads):
        base_offset = head_idx * batch_size * seq_len + pid_batch * seq_len
        vals = tl.load(tmp7_ptr + base_offset + offsets, mask=mask, other=0, eviction_policy="evict_last")
        head_max = tl.max(vals)
        max_val = tl.maximum(max_val, head_max)
    
    # Apply +1 and -9
    result = max_val + 1 - 9
    
    # Store scalar result
    tl.store(result_ptr + pid_batch, result)


@torch.fx.wrap
def fused_max_arith_impl(tmp_7):
    B, S = tmp_7.shape[1], tmp_7.shape[2]
    heads = tmp_7.shape[0]
    device = tmp_7.device
    dtype = tmp_7.dtype
    
    # Output tensor with same dtype as input
    out_result = torch.empty((B,), dtype=dtype, device=device)
    
    # Grid: one program per batch element
    grid = (B,)
    BLOCK_SIZE = min(1024, 1 << (S - 1).bit_length())
    
    fused_max_arith_kernel[grid](
        tmp_7, out_result,
        B, S, heads,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return out_result


def pattern(tmp_7):
    max_1 = tmp_7.max(0, keepdim=False)
    tmp_9 = max_1[0]
    max_2 = tmp_9.max(-1, keepdim=True)
    tmp_11 = max_2[0]
    tmp_12 = tmp_11 + 1
    tmp_13 = tmp_12 - 9
    return tmp_13


def replacement_args(tmp_7):
    return (tmp_7,)


def replacement_func():
    return fused_max_arith_impl