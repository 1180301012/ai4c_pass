import triton
import triton.language as tl


@triton.jit
def causal_attn_mask_kernel(
    attn_mask_ptr,   # pointer to in_0, shape [1, N], dtype int64, contiguous
    out_ptr,         # pointer to output, shape [1, 1, N, N], dtype float32, contiguous
    N: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """
    Each program handles one row `i` of the (N x N) output.
    out[0, 0, i, j] = 0.0  if (j <= i) AND (attn_mask[0, j] != 0)
                    = NEG_INF  otherwise
    """
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    col_mask = cols < N

    # Load attention mask values: in_0[0, cols]
    # in_0 is shape [1, N] contiguous, so in_0[0, j] is at offset j
    attn_vals = tl.load(attn_mask_ptr + cols, mask=col_mask, other=0)

    # Causal mask: position (row, col) is valid when col <= row
    causal_valid = cols <= row

    # Attention mask: valid when attn_vals != 0
    attn_valid = (attn_vals != 0)

    # Both conditions must hold
    is_valid = causal_valid & attn_valid

    NEG_INF = -3.4028234663852886e+38
    out_vals = tl.where(is_valid, 0.0, NEG_INF)

    # Store to output[0, 0, row, cols] — flat offset is row*N + cols
    out_offset = row * N + cols
    tl.store(out_ptr + out_offset, out_vals.to(tl.float32), mask=col_mask)