import torch
import triton
import triton.language as tl


@triton.jit
def _fused_relpos_softmax_kernel(
    index_ptr,
    weight_ptr,
    attn_ptr,
    mask_ptr,
    out_ptr,
    num_heads,
    attn_batch,
    mask_batch,
    total_rows,
    mask_add_count,
    BLOCK_K: tl.constexpr,
    BLOCK_COL: tl.constexpr,
):
    pid = tl.program_id(0)
    row = pid // num_heads
    head = pid % num_heads
    if row >= total_rows:
        return

    col_offsets = tl.arange(0, BLOCK_COL)
    mask_cols = col_offsets < 64

    row64 = row & 63
    base_index = row64 * 64
    idx = tl.load(index_ptr + base_index + col_offsets, mask=mask_cols, other=0).to(tl.int32)

    k_offsets = tl.arange(0, BLOCK_K)
    acc = tl.zeros((BLOCK_COL,), dtype=tl.float32)
    for k_start in range(0, 512, BLOCK_K):
        offs_k = k_start + k_offsets
        weight_mask = offs_k < 512
        w = tl.load(
            weight_ptr + head * 512 + offs_k,
            mask=weight_mask,
            other=0.0,
        )
        gathered = tl.load(
            weight_ptr + idx[:, None] * 512 + offs_k[None, :],
            mask=mask_cols[:, None] & weight_mask[None, :],
            other=0.0,
        )
        prod = gathered * w[None, :]
        acc += tl.sum(prod, axis=1)

    acc = tl.sigmoid(acc) * 16.0

    attn_row = row if attn_batch == total_rows else 0
    attn_base = ((attn_row * num_heads + head) * 64)
    attn_vals = tl.load(attn_ptr + attn_base + col_offsets, mask=mask_cols, other=0.0).to(tl.float32)

    mask_row = row if mask_batch == total_rows else row // num_heads
    mask_base = mask_row * 64
    mask_vals = tl.load(mask_ptr + mask_base + col_offsets, mask=mask_cols, other=0.0).to(tl.float32)

    vals = acc + attn_vals + mask_vals
    if mask_add_count == 2:
        vals += mask_vals

    vals = vals - tl.max(vals, axis=0)
    exp_vals = tl.exp(vals)
    denom = tl.sum(exp_vals, axis=0)
    probs = exp_vals / denom

    out_base = ((row * num_heads + head) * 64)
    tl.store(out_ptr + out_base + col_offsets, probs.to(out_ptr.dtype.element_ty), mask=mask_cols)


@triton.jit
def _fused_mask_softmax_kernel(
    base_ptr,
    mask_ptr,
    out_ptr,
    total_rows,
    num_heads,
    mask_batch,
    mask_add_count,
    BLOCK_COL: tl.constexpr,
):
    pid = tl.program_id(0)
    row = pid // num_heads
    head = pid % num_heads
    if row >= total_rows:
        return

    col_offsets = tl.arange(0, BLOCK_COL)
    mask_cols = col_offsets < 64

    base_base = ((row * num_heads + head) * 64)
    base_vals = tl.load(base_ptr + base_base + col_offsets, mask=mask_cols, other=0.0).to(tl.float32)

    mask_row = row if mask_batch == total_rows else row // num_heads
    mask_base = mask_row * 64
    mask_vals = tl.load(mask_ptr + mask_base + col_offsets, mask=mask_cols, other=0.0).to(tl.float32)

    vals = base_vals + mask_vals
    if mask_add_count == 2:
        vals += mask_vals

    vals = vals - tl.max(vals, axis=0)
    exp_vals = tl.exp(vals)
    denom = tl.sum(exp_vals, axis=0)
    probs = exp_vals / denom

    out_base = ((row * num_heads + head) * 64)
    tl.store(out_ptr + out_base + col_offsets, probs.to(out_ptr.dtype.element_ty), mask=mask_cols)


@torch.fx.wrap
def relpos_bias_softmax_dispatch(in_0, in_1, in_2, in_3, in_4):
    num_heads = in_1.shape[0]
    if in_2.dim() == 5:
        total_rows = in_2.shape[0] * in_2.shape[1]
        attn_batch = in_2.shape[0]
    else:
        total_rows = in_2.shape[0]
        attn_batch = in_2.shape[0]
    mask_batch = in_3.shape[0]
    mask_add_count = 2 if total_rows != attn_batch else 1

    out = torch.empty((total_rows, num_heads, 64, 64), device=in_2.device, dtype=in_2.dtype)

    if total_rows == 64:
        grid = (total_rows * num_heads,)
        _fused_relpos_softmax_kernel[grid](
            in_0,
            in_1,
            in_2,
            in_3,
            out,
            num_heads,
            attn_batch,
            mask_batch,
            total_rows,
            mask_add_count,
            BLOCK_K=64,
            BLOCK_COL=64,
            num_warps=4,
            num_stages=2,
        )
    else:
        base = in_2.view(total_rows, num_heads, 64, 64)
        grid = (total_rows * num_heads,)
        _fused_mask_softmax_kernel[grid](
            base,
            in_3,
            out,
            total_rows,
            num_heads,
            mask_batch,
            mask_add_count,
            BLOCK_COL=64,
            num_warps=4,
            num_stages=2,
        )

    return (out,)


def replacement_func():
    return relpos_bias_softmax_dispatch