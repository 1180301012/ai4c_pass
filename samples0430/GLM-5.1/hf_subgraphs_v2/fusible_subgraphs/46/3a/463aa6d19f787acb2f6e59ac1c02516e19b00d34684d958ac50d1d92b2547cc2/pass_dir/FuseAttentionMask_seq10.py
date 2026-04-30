import torch
import triton
import triton.language as tl
from torch import device

NEG_INF = -3.4028234663852886e+38


def pattern(in_0):
    tmp_1 = torch.arange(0, 10, device=device(type='cuda', index=0))
    tmp_2 = torch.full((10, 10), fill_value=-3.4028234663852886e+38, dtype=torch.float32, device=device(type='cuda', index=0))
    tmp_3 = torch.triu(tmp_2, diagonal=1)
    tmp_4 = torch.arange(10, device=device(type='cuda', index=0))
    tmp_5 = tmp_1.reshape(-1, 1)
    tmp_6 = tmp_4 > tmp_5
    tmp_3 *= tmp_6
    tmp_7 = tmp_3
    tmp_8 = tmp_7[(None, None, slice(None, None, None), slice(None, None, None))]
    tmp_9 = tmp_8.expand(1, 1, -1, -1)
    tmp_10 = tmp_9.clone()
    tmp_11 = tmp_10[(slice(None, None, None), slice(None, None, None), slice(None, None, None), slice(None, 10, None))]
    tmp_12 = in_0[(slice(None, None, None), None, None, slice(None, None, None))]
    tmp_13 = tmp_12.to(device(type='cuda', index=0))
    tmp_14 = tmp_11 + tmp_13
    tmp_15 = tmp_14.__eq__(0)
    tmp_16 = tmp_10[(slice(None, None, None), slice(None, None, None), slice(None, None, None), slice(None, 10, None))]
    tmp_17 = tmp_16.masked_fill(tmp_15, -3.4028234663852886e+38)
    tmp_10[(slice(None, None, None), slice(None, None, None), slice(None, None, None), slice(None, 10, None))] = tmp_17
    tmp_19 = tmp_10.__eq__(-3.4028234663852886e+38)
    tmp_20 = torch.all(tmp_19, dim=-1, keepdim=True)
    tmp_21 = ~tmp_20
    tmp_22 = tmp_10.mul(tmp_21)
    return (tmp_22,)


def replacement_args(in_0):
    return (in_0, "seq10")


@triton.jit
def fused_attention_mask_kernel(
    mask_ptr,
    out_ptr,
    seq_len,
    mask_batch_stride,
    mask_seq_stride,
    out_batch_stride,
    out_row_stride,
    out_col_stride,
    NEG_INF_VAL: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    batch_idx = pid // seq_len
    row_idx = pid % seq_len

    offsets = tl.arange(0, BLOCK_SIZE)
    valid = offsets < seq_len

    # Load attention mask for this batch (int64 values)
    mask_vals = tl.load(mask_ptr + batch_idx * mask_batch_stride + offsets * mask_seq_stride,
                        mask=valid, other=0)

    # Check if row has any valid token (mask[j] != 0 for some j <= row_idx)
    valid_check = offsets <= row_idx
    row_has_valid = tl.sum((mask_vals != 0) & valid_check & valid) > 0

    # Compute output for each column
    causal_masked = offsets > row_idx
    padding_masked = mask_vals == 0
    is_blocked = causal_masked | padding_masked

    # Output: -inf where blocked (if row has valid positions), 0 otherwise
    out_vals = tl.where(row_has_valid & is_blocked & valid, NEG_INF_VAL, 0.0)

    # Store
    out_base = batch_idx * out_batch_stride + row_idx * out_row_stride
    tl.store(out_ptr + out_base + offsets * out_col_stride, out_vals, mask=valid)


@triton.jit
def mask_finalize_kernel(
    mask_ptr,
    out_ptr,
    n_rows,
    seq_len,
    mask_stride_0,
    mask_stride_2,
    mask_stride_3,
    out_stride_0,
    out_stride_2,
    out_stride_3,
    NEG_INF_VAL: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    batch_idx = pid // seq_len
    row_idx = pid % seq_len

    offsets = tl.arange(0, BLOCK_SIZE)
    valid = offsets < seq_len

    # Load row from mask
    row_base = batch_idx * mask_stride_0 + row_idx * mask_stride_2
    row_vals = tl.load(mask_ptr + row_base + offsets * mask_stride_3,
                       mask=valid, other=NEG_INF_VAL)

    # Check if any value in row is NOT -inf (meaning row has valid attention positions)
    any_not_inf = tl.sum((row_vals != NEG_INF_VAL) & valid) > 0

    # Output: keep original values if row has valid positions, zero out if all -inf
    out_vals = tl.where(any_not_inf, row_vals, 0.0)

    # Store
    out_base = batch_idx * out_stride_0 + row_idx * out_stride_2
    tl.store(out_ptr + out_base + offsets * out_stride_3, out_vals, mask=valid)


def _fused_impl(attention_mask):
    batch_size = attention_mask.shape[0]
    seq_len = attention_mask.shape[-1]

    out = torch.empty(batch_size, 1, seq_len, seq_len, dtype=torch.float32, device=attention_mask.device)

    BLOCK_SIZE = triton.next_power_of_2(seq_len)

    grid = (batch_size * seq_len,)

    fused_attention_mask_kernel[grid](
        mask_ptr=attention_mask,
        out_ptr=out,
        seq_len=seq_len,
        mask_batch_stride=attention_mask.stride(0),
        mask_seq_stride=attention_mask.stride(1),
        out_batch_stride=out.stride(0),
        out_row_stride=out.stride(2),
        out_col_stride=out.stride(3),
        NEG_INF_VAL=NEG_INF,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    return out


def _mask_finalize_impl(mask_tensor):
    batch_size = mask_tensor.shape[0]
    seq_len = mask_tensor.shape[-1]

    out = torch.empty_like(mask_tensor)

    n_rows = batch_size * seq_len
    BLOCK_SIZE = triton.next_power_of_2(seq_len)

    grid = (n_rows,)

    mask_finalize_kernel[grid](
        mask_ptr=mask_tensor,
        out_ptr=out,
        n_rows=n_rows,
        seq_len=seq_len,
        mask_stride_0=mask_tensor.stride(0),
        mask_stride_2=mask_tensor.stride(2),
        mask_stride_3=mask_tensor.stride(3),
        out_stride_0=out.stride(0),
        out_stride_2=out.stride(2),
        out_stride_3=out.stride(3),
        NEG_INF_VAL=NEG_INF,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    return out


@torch.fx.wrap
def attention_mask_dispatch(*args):
    route_str = args[-1]
    if route_str == "seq21":
        attention_mask = args[0]
        return _fused_impl(attention_mask)
    elif route_str == "seq10":
        attention_mask = args[0]
        return _fused_impl(attention_mask)
    elif route_str == "seq13":
        attention_mask = args[0]
        return _fused_impl(attention_mask)
    elif route_str == "finalize":
        mask_tensor = args[0]
        return _mask_finalize_impl(mask_tensor)
    else:
        raise ValueError(f"Unknown route: {route_str}")


def replacement_func():
    return attention_mask_dispatch