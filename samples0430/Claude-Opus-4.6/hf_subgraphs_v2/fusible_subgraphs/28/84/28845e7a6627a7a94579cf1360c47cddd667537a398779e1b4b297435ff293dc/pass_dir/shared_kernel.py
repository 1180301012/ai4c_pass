import torch
import triton
import triton.language as tl


@triton.jit
def fused_add_softmax_kernel(
    in_0_ptr,
    in_1_ptr,
    out_ptr,
    H,
    S,
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0)
    head_idx = row_idx // S
    seq_idx = row_idx % S

    col_offsets = tl.arange(0, BLOCK_SIZE)
    valid_mask = col_offsets < S

    # Load in_0 [1, 1, S, S] - broadcast over heads
    in_0_offset = seq_idx * S + col_offsets
    in_0_vals = tl.load(in_0_ptr + in_0_offset, mask=valid_mask, other=0.0)

    # Load in_1 [1, H, S, S]
    in_1_offset = head_idx * S * S + seq_idx * S + col_offsets
    in_1_vals = tl.load(in_1_ptr + in_1_offset, mask=valid_mask, other=0.0)

    # Add in native dtype (matches PyTorch behavior)
    vals = in_1_vals + in_0_vals

    # Promote to float32 and clamp at -FLT_MAX
    vals_f32 = vals.to(tl.float32)
    vals_f32 = tl.maximum(vals_f32, -3.4028234663852886e+38)

    # Set padding positions to -inf so they don't affect softmax
    vals_f32 = tl.where(valid_mask, vals_f32, float('-inf'))

    # Numerically stable softmax
    max_val = tl.max(vals_f32, axis=0)
    vals_f32 = vals_f32 - max_val
    exp_vals = tl.exp(vals_f32)
    sum_exp = tl.sum(exp_vals, axis=0)
    softmax_vals = exp_vals / sum_exp

    # Store as float32
    out_offset = row_idx * S + col_offsets
    tl.store(out_ptr + out_offset, softmax_vals, mask=valid_mask)


@torch.fx.wrap
def fused_add_softmax(in_0, in_1):
    # Determine shapes from in_1 [1, H, S, S]
    shape = in_1.shape
    H = shape[1]
    S = shape[2]

    # Output shape is [H, S, S] with float32 dtype (due to type promotion in original)
    out = torch.empty((H, S, S), dtype=torch.float32, device=in_1.device)

    # BLOCK_SIZE: next power of 2 >= S
    if S <= 16:
        BLOCK_SIZE = 16
    elif S <= 32:
        BLOCK_SIZE = 32
    else:
        BLOCK_SIZE = 64

    # Launch kernel: one program per row of softmax
    num_rows = H * S
    fused_add_softmax_kernel[(num_rows,)](
        in_0, in_1, out,
        H, S,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    return out