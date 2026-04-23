import torch
import triton
import triton.language as tl
from torch import device

# Pattern matching for float32/seq_len=13 graph
# Rewritten with out-of-place equivalents of in-place operations to avoid FX dead code
def pattern(in_0):
    tmp_1 = torch.arange(0, 13, device=device(type='cuda', index=0))
    tmp_2 = torch.full((13, 13), fill_value=-3.4028234663852886e+38, dtype=torch.float32, device=device(type='cuda', index=0))
    tmp_3 = torch.triu(tmp_2, diagonal=1)
    tmp_4 = torch.arange(13, device=device(type='cuda', index=0))
    tmp_5 = tmp_1.reshape(-1, 1)
    tmp_6 = tmp_4 > tmp_5
    tmp_7 = tmp_3 * tmp_6
    tmp_8 = tmp_7[None, None, slice(None, None, None), slice(None, None, None)]
    tmp_9 = tmp_8.expand(1, 1, -1, -1)
    tmp_10 = tmp_9.clone()
    tmp_11 = tmp_10[slice(None, None, None), slice(None, None, None), slice(None, None, None), slice(None, 13, None)]
    tmp_12 = in_0[slice(None, None, None), None, None, slice(None, None, None)]
    tmp_13 = tmp_12.to(device(type='cuda', index=0))
    tmp_14 = tmp_11 + tmp_13
    tmp_15 = tmp_14.__eq__(0)
    tmp_16 = tmp_10[slice(None, None, None), slice(None, None, None), slice(None, None, None), slice(None, 13, None)]
    tmp_17 = tmp_16.masked_fill(tmp_15, -3.4028234663852886e+38)
    tmp_19 = tmp_17.__eq__(-3.4028234663852886e+38)
    tmp_20 = torch.all(tmp_19, dim=-1, keepdim=True)
    tmp_21 = ~tmp_20
    tmp_22 = tmp_17.mul(tmp_21)
    return (tmp_22,)

def replacement_args(in_0):
    return (in_0, "route_fp32_13")


@triton.jit
def fused_attention_mask_kernel(
    in_ptr,
    out_ptr,
    batch_size,
    seq_len,
    NEG_INF: tl.constexpr,
    BLOCK_COLS: tl.constexpr,
):
    pid = tl.program_id(0)
    total_rows = batch_size * seq_len
    if pid >= total_rows:
        return

    batch_idx = pid // seq_len
    row_idx = pid % seq_len

    col_range = tl.arange(0, BLOCK_COLS)
    col_mask = col_range < seq_len

    # Load attention mask for this batch element
    mask_offsets = batch_idx * seq_len + col_range
    attn_vals = tl.load(in_ptr + mask_offsets, mask=col_mask, other=1)

    # Compute causal mask: j > row_idx means future position (masked)
    causal = col_range > row_idx

    # Compute padding mask: attention_mask[j] == 0 means padded (masked)
    padded = attn_vals == 0

    # Combined: -inf where causal or padded
    is_inf = causal | (padded & col_mask)
    row_vals = tl.where(is_inf, NEG_INF, 0.0)

    # Check if all valid columns are -inf (entire row masked)
    valid_inf = (row_vals == NEG_INF) & col_mask
    all_inf_count = tl.sum(valid_inf.to(tl.int32))
    all_inf = all_inf_count == seq_len

    # Zero out row if all positions are -inf
    if all_inf:
        row_vals = tl.where(col_mask, 0.0, row_vals)

    # Store output: [batch, 1, seq_len, seq_len]
    out_offsets = (batch_idx * seq_len + row_idx) * seq_len + col_range
    tl.store(out_ptr + out_offsets, row_vals, mask=col_mask)


@torch.fx.wrap
def fused_attention_mask_dispatch(in_0, route):
    NEG_INF = -3.4028234663852886e+38

    batch_size = in_0.shape[0]
    seq_len = in_0.shape[1]

    # Output shape: [batch, 1, seq_len, seq_len]
    out = torch.full((batch_size, 1, seq_len, seq_len), NEG_INF, dtype=torch.float32, device=in_0.device)

    BLOCK_COLS = triton.next_power_of_2(seq_len)

    total_rows = batch_size * seq_len
    grid = (total_rows,)

    fused_attention_mask_kernel[grid](
        in_ptr=in_0,
        out_ptr=out,
        batch_size=batch_size,
        seq_len=seq_len,
        NEG_INF=NEG_INF,
        BLOCK_COLS=BLOCK_COLS,
    )

    return out


def replacement_func():
    return fused_attention_mask_dispatch