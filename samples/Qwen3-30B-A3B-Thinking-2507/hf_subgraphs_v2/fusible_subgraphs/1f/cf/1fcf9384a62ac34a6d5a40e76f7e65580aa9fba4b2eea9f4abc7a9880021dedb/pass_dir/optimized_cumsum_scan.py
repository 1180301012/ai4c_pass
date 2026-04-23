import torch
import triton
import triton.language as tl

# Pattern matching function
def pattern(in_0):
    tmp_1 = in_0.ne(1)
    tmp_2 = tmp_1.int()
    tmp_3 = torch.cumsum(tmp_2, dim=1)
    tmp_4 = tmp_3.type_as(tmp_2)
    tmp_5 = tmp_4 + 0
    tmp_6 = tmp_5 * tmp_2
    tmp_7 = tmp_6.long()
    tmp_8 = tmp_7 + 1
    return tmp_8

# Argument extraction function
def replacement_args(in_0):
    return (in_0,)

# Triton kernel for optimized cumulative sum scan
@triton.jit
def cumsum_scan_kernel(
    in_ptr,
    out_ptr,
    B,
    S,
    BLOCK_SIZE: tl.constexpr,
):
    # Process one batch row per block
    batch_id = tl.program_id(0)
    start = batch_id * S
    offsets = start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < (batch_id + 1) * S

    # Load input and compute mask
    x = tl.load(in_ptr + offsets, mask=mask)
    mask_val = (x != 1).to(tl.int64)

    # Compute inclusive scan on mask_val
    cumsum = mask_val
    for i in range(0, 11):  # Handles S up to 2^10 = 1024
        step = 1 << i
        prev_idx = offsets - step
        prev_mask = offsets >= step
        prev_val = tl.load(in_ptr + prev_idx, mask=prev_mask, other=0)
        cumsum = cumsum + prev_val

    # Apply final operations: multiply by mask, add 1, convert to long
    result = cumsum * mask_val + 1
    tl.store(out_ptr + offsets, result, mask=mask)

# Kernel wrapper
def optimized_kernel_wrapper(in_0):
    B, S = in_0.shape
    out = torch.empty_like(in_0, dtype=torch.int64)

    BLOCK_SIZE = 128
    num_blocks = B

    cumsum_scan_kernel[(num_blocks,)](
        in_ptr=in_0,
        out_ptr=out,
        B=B,
        S=S,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return out

# Replacement function
def replacement_func():
    return optimized_kernel_wrapper