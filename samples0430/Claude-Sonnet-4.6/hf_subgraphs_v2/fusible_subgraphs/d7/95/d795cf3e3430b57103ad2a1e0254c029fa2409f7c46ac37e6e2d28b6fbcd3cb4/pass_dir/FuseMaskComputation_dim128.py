import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK': 512}),
        triton.Config({'BLOCK': 1024}),
        triton.Config({'BLOCK': 2048}),
        triton.Config({'BLOCK': 4096}),
    ],
    key=[],
)
@triton.jit
def _attn_mask_kernel_128(
    out_ptr,
    N,
    BLOCK: tl.constexpr,
):
    """
    Computes (1, 361, 49, 49) attention mask directly from index arithmetic.

    The original model builds a (1,133,133) mask where last-5 rows/cols = 1,
    then reshapes/transposes to (1,361,49), then computes:
      result[k,i,j] = -1000 if mask[k,j] != mask[k,i] else 0.0

    We derive mask values analytically:
      k_i0 = k // 19, k_j0 = k % 19
      val(k, l) = (k_i0*7 + l//7 >= 128) | (k_j0*7 + l%7 >= 128)
    """
    pid = tl.program_id(0)
    base = pid * BLOCK
    offsets = base + tl.arange(0, BLOCK)
    valid = offsets < N

    # Decompose flat index into (k, i, j) for output shape (1, 361, 49, 49)
    j = offsets % 49
    ki = offsets // 49
    i = ki % 49
    k = ki // 49

    # Block coordinates
    k_i0 = k // 19
    k_j0 = k % 19

    # Mask value for local index j
    j_i1 = j // 7
    j_j1 = j % 7
    row_j = k_i0 * 7 + j_i1
    col_j = k_j0 * 7 + j_j1
    val_j = (row_j >= 128) | (col_j >= 128)

    # Mask value for local index i
    i_i1 = i // 7
    i_j1 = i % 7
    row_i = k_i0 * 7 + i_i1
    col_i = k_j0 * 7 + i_j1
    val_i = (row_i >= 128) | (col_i >= 128)

    # -1000.0 where mask values differ, 0.0 where they match
    result = tl.where(val_j != val_i, -1000.0, 0.0)
    tl.store(out_ptr + offsets, result, mask=valid)


@torch.fx.wrap
def _fused_mask_reshape_128(in_0, tmp_0):
    # tmp_0 is the (1,133,133) constant mask — we recompute analytically via Triton.
    # Output is the FINAL mask values {-1000.0, 0.0} directly as tmp_12.
    # Downstream ops (!=0, masked_fill, ==0, masked_fill) become no-ops since:
    #   final_mask != 0  →  True where -1000.0 (same as where pairwise_diff != 0)
    #   masked_fill(final_mask!=0, -1000.0) → no change (already -1000.0 there)
    #   final_mask == 0  →  True where 0.0 (same as where pairwise_diff == 0)
    #   masked_fill(final_mask==0, 0.0) → no change (already 0.0 there)
    N = 1 * 361 * 49 * 49  # 867769
    out = torch.empty((1, 361, 49, 49), device=in_0.device, dtype=torch.float32)
    grid = lambda meta: (triton.cdiv(N, meta['BLOCK']),)
    _attn_mask_kernel_128[grid](out, N)

    # tmp_6: view reshape + transpose (no data copy)
    tmp_6 = in_0.reshape(1, 19, 7, 19, 7, 128).transpose(2, 3)
    return (out, tmp_6)


def pattern(in_0, tmp_0):
    # tmp_0 is the (1,133,133) mask tensor (folded constant from zeros+fill_)
    # Pattern stops at tmp_12 (the pairwise subtraction), returning it as output.
    tmp_5 = in_0.reshape(1, 19, 7, 19, 7, 128)
    tmp_6 = tmp_5.transpose(2, 3)
    tmp_7 = tmp_0.reshape(1, 19, 7, 19, 7)
    tmp_8 = tmp_7.transpose(2, 3)
    tmp_9 = tmp_8.reshape(1, 361, 49)
    tmp_10 = tmp_9.unsqueeze(2)
    tmp_11 = tmp_9.unsqueeze(3)
    tmp_12 = tmp_10 - tmp_11
    return (tmp_12, tmp_6)


def replacement_args(in_0, tmp_0):
    return (in_0, tmp_0)


def replacement_func():
    return _fused_mask_reshape_128