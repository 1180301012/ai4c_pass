import torch
import triton
import triton.language as tl


NEG_INF = -3.4028234663852886e+38


@triton.jit
def _fused_causal_attention_mask_kernel(
    attention_mask_ptr,
    out_ptr,
    SEQ_LEN: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row_ids = tl.arange(0, BLOCK)
    col_ids = tl.arange(0, BLOCK)

    valid_rows = row_ids < SEQ_LEN
    valid_cols = col_ids < SEQ_LEN
    store_mask = valid_rows[:, None] & valid_cols[None, :]

    # in_0 has shape [1, SEQ_LEN], so the first row is contiguous in memory.
    token_keep = tl.load(attention_mask_ptr + col_ids, mask=valid_cols, other=0)
    causal_keep = col_ids[None, :] <= row_ids[:, None]
    padding_keep = token_keep[None, :] != 0
    visible = causal_keep & padding_keep

    out = tl.where(visible, 0.0, NEG_INF)
    offsets = row_ids[:, None] * SEQ_LEN + col_ids[None, :]
    tl.store(out_ptr + offsets, out, mask=store_mask)


@torch.fx.wrap
def fused_attention_mask(in_0, seq_len):
    out = torch.empty((1, 1, seq_len, seq_len), device=in_0.device, dtype=torch.float32)

    if seq_len == 9:
        _fused_causal_attention_mask_kernel[(1,)](
            in_0,
            out,
            SEQ_LEN=9,
            BLOCK=16,
            num_warps=1,
        )
    elif seq_len == 13:
        _fused_causal_attention_mask_kernel[(1,)](
            in_0,
            out,
            SEQ_LEN=13,
            BLOCK=16,
            num_warps=1,
        )
    else:
        # Generic fallback for unexpected sequence lengths.
        block = triton.next_power_of_2(seq_len)
        block = 16 if block < 16 else block
        _fused_causal_attention_mask_kernel[(1,)](
            in_0,
            out,
            SEQ_LEN=seq_len,
            BLOCK=block,
            num_warps=1,
        )

    return out