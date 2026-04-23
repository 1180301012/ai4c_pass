import torch
import triton
import triton.language as tl


def pattern(tmp_15):
    tmp_16 = tmp_15 < 0
    tmp_17 = tmp_16.to(torch.int64)
    tmp_18 = tmp_17 * 16
    tmp_19 = 0 + tmp_18
    tmp_20 = torch.abs(tmp_15)
    tmp_21 = tmp_20 < 8
    tmp_22 = tmp_20.float()
    tmp_23 = tmp_22 / 8
    tmp_24 = torch.log(tmp_23)
    tmp_25 = tmp_24 / 2.772588722239781
    tmp_26 = tmp_25 * 8
    tmp_27 = tmp_26.to(torch.int64)
    tmp_28 = 8 + tmp_27
    tmp_29 = torch.full_like(tmp_28, 15)
    tmp_30 = torch.min(tmp_28, tmp_29)
    tmp_31 = torch.where(tmp_21, tmp_20, tmp_30)
    tmp_32 = tmp_19 + tmp_31
    return tmp_32


def replacement_args(tmp_15):
    return (tmp_15,)


@triton.jit
def position_encoding_kernel(
    tmp_15_ptr, out_ptr,
    n_rows, n_cols,
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0)
    col_block = tl.program_id(1)
    col_start = col_block * BLOCK_SIZE
    col_offsets = col_start + tl.arange(0, BLOCK_SIZE)

    mask = (row_idx < n_rows) & (col_offsets < n_cols)

    ptr_offsets = row_idx * n_cols + col_offsets
    vals = tl.load(tmp_15_ptr + ptr_offsets, mask=mask, other=0)

    # Position encoding computation
    # tmp_16 = tmp_15 < 0 → base = 16 if negative, 0 otherwise
    is_neg = vals < 0
    base = tl.where(is_neg, 16, 0)

    # tmp_20 = abs(tmp_15)
    abs_val = tl.abs(vals)

    # tmp_21 = tmp_20 < 8
    is_close = abs_val < 8

    # Logarithmic bucketing for far positions (abs_val >= 8)
    # tmp_22 = tmp_20.float()
    # tmp_23 = tmp_22 / 8
    # tmp_24 = torch.log(tmp_23)
    # tmp_25 = tmp_24 / 2.772588722239781  (ln(16))
    # tmp_26 = tmp_25 * 8
    # tmp_27 = tmp_26.to(int64)
    # tmp_28 = 8 + tmp_27
    # tmp_30 = min(tmp_28, 15)
    abs_val_f = abs_val.to(tl.float32)
    # Use safe value (8.0) for close positions to avoid log(0)
    safe_abs = tl.where(is_close, 8.0, abs_val_f)
    ratio = safe_abs / 8.0
    log_ratio = tl.log(ratio)
    log16_ratio = log_ratio / 2.772588722239781
    bucket_f = 8.0 * log16_ratio
    bucket_i = bucket_f.to(tl.int64)
    far_offset = 8 + bucket_i
    # min(far_offset, 15)
    far_offset = tl.where(far_offset > 15, 15, far_offset)

    # tmp_31 = where(tmp_21, tmp_20, tmp_30)
    offset = tl.where(is_close, abs_val, far_offset)

    # tmp_32 = tmp_19 + tmp_31 = base + offset
    result = base + offset

    # Cast to int64 for output
    result = result.to(tl.int64)

    tl.store(out_ptr + ptr_offsets, result, mask=mask)


@torch.fx.wrap
def position_encoding_wrapper(tmp_15):
    n_rows, n_cols = tmp_15.shape
    out = torch.empty((n_rows, n_cols), dtype=torch.int64, device=tmp_15.device)
    BLOCK_SIZE = 128
    grid = (n_rows, triton.cdiv(n_cols, BLOCK_SIZE))
    position_encoding_kernel[grid](
        tmp_15_ptr=tmp_15,
        out_ptr=out,
        n_rows=n_rows,
        n_cols=n_cols,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return out


def replacement_func():
    return position_encoding_wrapper