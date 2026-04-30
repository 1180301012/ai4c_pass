import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 16}, num_warps=1),
        triton.Config({'BLOCK_SIZE': 16}, num_warps=2),
        triton.Config({'BLOCK_SIZE': 32}, num_warps=1),
        triton.Config({'BLOCK_SIZE': 32}, num_warps=2),
    ],
    key=['N_COLS'],
)
@triton.jit
def fused_add_clamp_softmax_kernel(
    x_ptr,
    mask_ptr,
    out_ptr,
    stride_x_row,
    seq_len,
    stride_out_row,
    n_rows,
    N_COLS,
    NEG_INF: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    row_id = tl.program_id(0)

    cols = tl.arange(0, BLOCK_SIZE)
    col_mask = cols < N_COLS

    x_row_ptr = x_ptr + row_id * stride_x_row
    # expanded_attn_mask has shape [1,1,S,S]; after broadcasting every head uses the same row.
    # row_id enumerates flattened [H, S] rows, so the source row in the mask is row_id % S.
    mask_row_idx = row_id % seq_len
    mask_row_ptr = mask_ptr + mask_row_idx * seq_len
    out_row_ptr = out_ptr + row_id * stride_out_row

    x = tl.load(x_row_ptr + cols, mask=col_mask, other=0.0).to(tl.float32)
    m = tl.load(mask_row_ptr + cols, mask=col_mask, other=0.0).to(tl.float32)
    v = x + m
    neg_inf = tl.full((BLOCK_SIZE,), NEG_INF, tl.float32)
    v = tl.maximum(v, neg_inf)
    v = tl.where(col_mask, v, neg_inf)

    row_max = tl.max(v, axis=0)
    shifted = v - row_max
    numer = tl.where(col_mask, tl.exp(shifted), 0.0)
    denom = tl.sum(numer, axis=0)
    out = numer / denom

    tl.store(out_row_ptr + cols, out, mask=col_mask)


@torch.fx.wrap
def fused_add_clamp_softmax(x, mask):
    # x: [1, H, S, S], mask: [1, 1, S, S]
    # output required by graph: view(H, S, S)
    H = x.shape[1]
    S = x.shape[-1]
    n_rows = H * S

    x2 = x.reshape(n_rows, S)
    mask2 = mask.reshape(S, S)

    out = torch.empty((n_rows, S), device=x.device, dtype=torch.float32)

    grid = (n_rows,)
    # We use the same negative finite constant as the original graph.
    fused_add_clamp_softmax_kernel[grid](
        x2,
        mask2,
        out,
        x2.stride(0),
        S,
        out.stride(0),
        n_rows,
        S,
        NEG_INF=-3.4028234663852886e+38,
    )

    return out.reshape(H, S, S)


def shared_replacement_func():
    return fused_add_clamp_softmax