import torch
import triton
import triton.language as tl


@triton.jit
def attention_mask_kernel(
    in_ptr,
    out_ptr,
    N: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Each block handles one row of the NxN output
    row_idx = tl.program_id(0)
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < N

    # Load attention mask values for each column
    attn_mask = tl.load(in_ptr + col_offsets, mask=mask, other=0)

    # Causal: col <= row means attend (lower triangular including diagonal)
    causal_valid = col_offsets <= row_idx

    # Attention: in_0[0, j] == 1 means attend
    attn_valid = attn_mask == 1

    # Combined: attend only where both conditions are met
    combined = causal_valid & attn_valid

    # Output: 0.0 where combined is True, -FLT_MAX otherwise
    result = tl.where(combined, 0.0, -3.4028234663852886e+38)

    # Store to output [1, 1, N, N], flattened row-major
    out_offset = row_idx * N + col_offsets
    tl.store(out_ptr + out_offset, result, mask=mask)


@torch.fx.wrap
def fused_attention_mask_dispatch(in_0, N_val):
    N = int(N_val)
    out = torch.empty((1, 1, N, N), dtype=torch.float32, device=in_0.device)
    attention_mask_kernel[(N,)](
        in_0,
        out,
        N=N,
        BLOCK_SIZE=16,
    )
    return out