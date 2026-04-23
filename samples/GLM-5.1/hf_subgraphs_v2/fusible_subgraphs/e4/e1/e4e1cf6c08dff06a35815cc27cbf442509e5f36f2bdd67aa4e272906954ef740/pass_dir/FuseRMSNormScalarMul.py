import torch
import triton
import triton.language as tl


def pattern(tmp_2, in_1):
    tmp_4 = tmp_2.float()
    tmp_5 = tmp_4.pow(2)
    tmp_6 = tmp_5.mean(-1, keepdim=True)
    tmp_7 = tmp_6 + 1e-06
    tmp_8 = torch.rsqrt(tmp_7)
    tmp_9 = tmp_4 * tmp_8
    tmp_10 = in_1.float()
    tmp_11 = 1.0 + tmp_10
    tmp_12 = tmp_9 * tmp_11
    tmp_13 = tmp_12.type_as(tmp_2)
    return tmp_13


def replacement_args(tmp_2, in_1):
    return (tmp_2, in_1)


@triton.jit
def rmsnorm_kernel(
    in0_ptr,
    in1_ptr,
    out_ptr,
    n_rows,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0)
    if row_idx >= n_rows:
        return

    row_start = row_idx * n_cols
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols

    # Load input row (bfloat16 → float32)
    in0 = tl.load(in0_ptr + row_start + col_offsets, mask=mask, other=0.0).to(tl.float32)

    # Compute variance = mean(in0^2)
    sq = in0 * in0
    variance = tl.sum(sq, axis=0) / n_cols

    # variance + eps (eps=1e-06)
    variance_eps = variance + 1e-06

    # rsqrt(variance_eps)
    inv_rms = tl.rsqrt(variance_eps)

    # normalize: in0 * inv_rms
    normalized = in0 * inv_rms

    # Load weight, compute (1 + weight)
    weight = tl.load(in1_ptr + col_offsets, mask=mask, other=0.0).to(tl.float32)
    weight_shifted = 1.0 + weight

    # Final: normalized * (1 + weight)
    result = normalized * weight_shifted

    # Store bfloat16 output
    tl.store(out_ptr + row_start + col_offsets, result.to(tl.bfloat16), mask=mask)


@torch.fx.wrap
def rmsnorm_impl(tmp_2, in_1):
    hidden_dim = tmp_2.shape[-1]
    n_rows = tmp_2.numel() // hidden_dim

    out = torch.empty_like(tmp_2)

    BLOCK_SIZE = triton.next_power_of_2(hidden_dim)
    grid = (n_rows,)

    rmsnorm_kernel[grid](
        in0_ptr=tmp_2,
        in1_ptr=in_1,
        out_ptr=out,
        n_rows=n_rows,
        n_cols=hidden_dim,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    return out


def replacement_func():
    return rmsnorm_impl