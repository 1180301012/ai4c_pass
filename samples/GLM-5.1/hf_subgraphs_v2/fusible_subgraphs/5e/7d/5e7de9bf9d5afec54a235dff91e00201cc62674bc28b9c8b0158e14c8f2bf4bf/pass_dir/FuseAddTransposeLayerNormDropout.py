import torch
import triton
import triton.language as tl


def pattern(in_0, in_1, tmp_6, tmp_7):
    tmp_8 = tmp_6 + tmp_7
    tmp_9 = tmp_8.transpose(1, 2)
    tmp_10 = torch.nn.functional.layer_norm(tmp_9, (768,), in_1, in_0, 1e-05)
    tmp_11 = torch.nn.functional.dropout(tmp_10, 0.1, False, False)
    return (tmp_11,)


def replacement_args(in_0, in_1, tmp_6, tmp_7):
    return (tmp_6, tmp_7, in_1, in_0)


@triton.jit
def fused_add_trans_ln_kernel(
    a_ptr, b_ptr, weight_ptr, bias_ptr, out_ptr,
    n_cols, seq_len,
    a_stride0, a_stride1, a_stride2,
    b_stride0, b_stride1, b_stride2,
    out_stride0, out_stride1, out_stride2,
    eps,
    BLOCK_COLS: tl.constexpr,
    ROWS_PER_PROGRAM: tl.constexpr,
):
    pid = tl.program_id(0)
    row_start = pid * ROWS_PER_PROGRAM
    row_end = min(row_start + ROWS_PER_PROGRAM, seq_len)

    col_offsets = tl.arange(0, BLOCK_COLS)
    col_mask = col_offsets < n_cols
    w_vals = tl.load(weight_ptr + col_offsets, mask=col_mask, other=0.0)
    b_vals_bias = tl.load(bias_ptr + col_offsets, mask=col_mask, other=0.0)

    for row_idx in tl.range(row_start, row_end, 1):
        a_vals = tl.load(a_ptr + col_offsets * a_stride1 + row_idx * a_stride2, mask=col_mask, other=0.0)
        b_vals_local = tl.load(b_ptr + col_offsets * b_stride1 + row_idx * b_stride2, mask=col_mask, other=0.0)

        sum_vals = a_vals + b_vals_local

        mean = tl.sum(sum_vals, axis=0) / n_cols
        diff = sum_vals - mean
        var = tl.sum(diff * diff, axis=0) / n_cols
        rstd = 1.0 / tl.sqrt(var + eps)
        normed = diff * rstd

        out_vals = normed * w_vals + b_vals_bias
        tl.store(out_ptr + row_idx * out_stride1 + col_offsets * out_stride2, out_vals, mask=col_mask)


@torch.fx.wrap
def fused_add_trans_layernorm(a, b, weight, bias, eps=1e-05):
    batch = a.shape[0]
    n_cols = a.shape[1]
    seq_len = a.shape[2]

    out = torch.empty((batch, seq_len, n_cols), dtype=a.dtype, device=a.device)

    a_stride0, a_stride1, a_stride2 = a.stride()
    b_stride0, b_stride1, b_stride2 = b.stride()
    out_stride0, out_stride1, out_stride2 = out.stride()

    BLOCK_COLS = 1024
    ROWS_PER_PROGRAM = 4
    n_programs = (seq_len + ROWS_PER_PROGRAM - 1) // ROWS_PER_PROGRAM

    fused_add_trans_ln_kernel[(n_programs,)](
        a, b, weight, bias, out,
        n_cols=n_cols, seq_len=seq_len,
        a_stride0=a_stride0, a_stride1=a_stride1, a_stride2=a_stride2,
        b_stride0=b_stride0, b_stride1=b_stride1, b_stride2=b_stride2,
        out_stride0=out_stride0, out_stride1=out_stride1, out_stride2=out_stride2,
        eps=eps,
        BLOCK_COLS=BLOCK_COLS,
        ROWS_PER_PROGRAM=ROWS_PER_PROGRAM,
        num_warps=4,
        num_stages=2,
    )

    return out


def replacement_func():
    return fused_add_trans_layernorm