import torch
import triton
import triton.language as tl


@triton.jit
def fused_add_softmax_kernel(
    in0_ptr, in1_ptr, out_ptr,
    N_rows, M_cols, N_in0_rows,
    BLOCK_SIZE: tl.constexpr,
):
    """Fused add + softmax kernel.
    
    Each program processes one row of the output tensor.
    in_0 broadcasts over the 'heads' dimension, so each row in the
    output maps to a unique row in in_1 and a shared row in in_0.
    
    Row r in the 3D view [num_heads, N, M]:
      - in_1 row: offset = r * M_cols (since in_1 is contiguous)
      - in_0 row: offset = (r % N_in0_rows) * M_cols (broadcasting)
    """
    row_idx = tl.program_id(0)
    if row_idx >= N_rows:
        return

    in1_row_base = row_idx * M_cols
    in0_row_base = (row_idx % N_in0_rows) * M_cols
    out_row_base = row_idx * M_cols

    # Single-pass softmax for rows that fit in BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < M_cols

    # Load input rows
    in0_vals = tl.load(in0_ptr + in0_row_base + offsets, mask=mask, other=0.0)
    in1_vals = tl.load(in1_ptr + in1_row_base + offsets, mask=mask, other=0.0)

    # Add with broadcasting (in_0 broadcasts over heads dimension)
    sum_vals = in0_vals + in1_vals

    # Softmax: numerical stability - find max
    sum_vals_masked = tl.where(mask, sum_vals, -float('inf'))
    row_max = tl.max(sum_vals_masked)

    # Compute exp(x - max) and sum
    exp_vals = tl.where(mask, tl.exp(sum_vals - row_max), 0.0)
    exp_sum = tl.sum(exp_vals)

    # Normalize
    softmax_vals = tl.where(mask, exp_vals / exp_sum, 0.0)

    # Store result
    tl.store(out_ptr + out_row_base + offsets, softmax_vals, mask=mask)


@torch.fx.wrap
def fused_add_softmax_dispatch(in_0, in_1, route):
    """Shared dispatch wrapper for fused add + softmax.
    
    Handles both shape variants via route string:
    - "route_8_300_625": for bfloat16/float16 variants (8, 300, 625)
    - "route_8_625_625": for float32 variant (8, 625, 625)
    
    Dimensions are extracted from input tensors at runtime.
    """
    # Extract dimensions from input shapes
    # in_0: [1, 1, N, M], in_1: [1, num_heads, N, M]
    N = in_0.shape[2]
    M = in_0.shape[3]
    num_heads = in_1.shape[1]
    N_rows = num_heads * N
    M_cols = M
    N_in0_rows = N

    BLOCK_SIZE = 1024

    # Allocate output with same shape and dtype as in_1
    out = torch.empty_like(in_1)

    if route == "route_8_300_625":
        grid = (N_rows,)
        fused_add_softmax_kernel[grid](
            in0_ptr=in_0, in1_ptr=in_1, out_ptr=out,
            N_rows=N_rows, M_cols=M_cols, N_in0_rows=N_in0_rows,
            BLOCK_SIZE=BLOCK_SIZE,
        )
    elif route == "route_8_625_625":
        grid = (N_rows,)
        fused_add_softmax_kernel[grid](
            in0_ptr=in_0, in1_ptr=in_1, out_ptr=out,
            N_rows=N_rows, M_cols=M_cols, N_in0_rows=N_in0_rows,
            BLOCK_SIZE=BLOCK_SIZE,
        )
    else:
        raise ValueError(f"Unknown route: {route}")

    # Return (tmp_5, tmp_3) matching original model output structure
    # tmp_5 = result in 3D view [num_heads, N, M]
    # tmp_3 = result in 4D view [1, num_heads, N, M]
    return (out.view(num_heads, N, M), out)