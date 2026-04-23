import torch
import triton
import triton.language as tl

# NOTE: Scratch buffer is needed because Triton's Kogge-Stone scan requires
# store/load of intermediate partial sums, which can only happen via global memory.

def pattern(in_0):
    tmp_0 = in_0
    tmp_1 = torch.cumsum(tmp_0, dim=1)
    tmp_2 = tmp_1 * tmp_0
    tmp_3 = tmp_2 - 1
    tmp_4 = tmp_3.long()
    tmp_5 = tmp_4[slice(None, None, None), slice(0, None, None)]
    tmp_6 = tmp_5 + 2
    return (tmp_6,)


def replacement_args(in_0):
    return (in_0,)


@triton.jit
def fused_scan_kernel(
    in_ptr,
    out_ptr,
    scratch_ptr,
    num_cols,
    BLOCK_SIZE: tl.constexpr,
):
    """Fused kernel using Kogge-Stone parallel scan algorithm.
    
    Computes: cumsum(in_0, dim=1) * in_0 + 1
    Simplified from: cumsum * x - 1 .long() [:] + 2 = cumsum * x + 1
    
    Each program processes one row. scratch_ptr is a zero-initialized buffer
    of size [num_rows, BLOCK_SIZE] for intermediate store/load in the scan.
    
    Kogge-Stone inclusive scan:
    - Store current partial sums to scratch
    - Load shifted values (offset by -stride) from scratch
    - Accumulate into partial sums
    - Repeat for strides 1, 2, 4, ... up to log2(BLOCK_SIZE)
    
    We hardcode 8 iterations (supports BLOCK_SIZE up to 256).
    For smaller BLOCK_SIZE, extra iterations have stride >= BLOCK_SIZE, so
    shifted_offsets are always negative, shifted_vals are 0 (no-op).
    """
    pid = tl.program_id(0)
    row_in_offset = pid * num_cols
    row_scratch_offset = pid * BLOCK_SIZE

    col_offsets = tl.arange(0, BLOCK_SIZE)
    in_offsets = row_in_offset + col_offsets
    scratch_offsets = row_scratch_offset + col_offsets
    mask = col_offsets < num_cols

    # Load input row (positions >= num_cols get 0 due to other=0)
    x = tl.load(in_ptr + in_offsets, mask=mask, other=0)

    # Initialize scan with input values
    scan_val = x

    # Kogge-Stone inclusive scan iterations
    # 8 iterations supports BLOCK_SIZE up to 256 (2^8)
    for d in tl.static_range(0, 8):
        stride = 1 << d  # strides: 1, 2, 4, 8, 16, 32, 64, 128
        # Store current scan values to scratch buffer
        tl.store(scratch_ptr + scratch_offsets, scan_val)
        # Load shifted values from scratch (shifted by -stride positions)
        shifted_offsets = col_offsets - stride
        shifted_scratch_offsets = row_scratch_offset + shifted_offsets
        shifted_mask = shifted_offsets >= 0
        shifted_vals = tl.load(scratch_ptr + shifted_scratch_offsets, mask=shifted_mask, other=0)
        # Accumulate: scan_val[i] += scan_val[i-stride] for i >= stride
        scan_val = scan_val + shifted_vals

    # After Kogge-Stone, scan_val contains the inclusive prefix sum
    # For positions >= num_cols, scan_val may have accumulated zero-padding sums
    # but those positions are masked out in the final store

    # Compute final result: cumsum * input + 1
    result = tl.where(mask, scan_val * x + 1, 0)

    # Store result to output
    out_offsets = row_in_offset + col_offsets
    tl.store(out_ptr + out_offsets, result, mask=mask)


@torch.fx.wrap
def fused_cumsum_mul_sub_cast_add(in_0):
    num_rows = in_0.shape[0]
    num_cols = in_0.shape[1]

    # Choose BLOCK_SIZE as next power of 2 >= num_cols
    BLOCK_SIZE = 1
    while BLOCK_SIZE < num_cols:
        BLOCK_SIZE *= 2

    out = torch.empty_like(in_0)
    # Scratch buffer: zero-initialized, used for Kogge-Stone intermediate store/load
    scratch = torch.zeros((num_rows, BLOCK_SIZE), dtype=in_0.dtype, device=in_0.device)

    # Launch one program per row
    grid = (num_rows,)

    fused_scan_kernel[grid](
        in_ptr=in_0,
        out_ptr=out,
        scratch_ptr=scratch,
        num_cols=num_cols,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    return (out,)


def replacement_func():
    return fused_cumsum_mul_sub_cast_add