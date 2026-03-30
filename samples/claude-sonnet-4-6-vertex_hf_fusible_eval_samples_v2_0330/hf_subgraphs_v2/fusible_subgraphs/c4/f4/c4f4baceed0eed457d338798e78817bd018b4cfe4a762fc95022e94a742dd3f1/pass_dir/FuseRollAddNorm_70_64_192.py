import torch
import triton
import triton.language as tl


def pattern(in_0, in_1, in_2, in_3):
    tmp_2 = in_3.contiguous()
    tmp_3 = tmp_2.view(-1, 70, 70, 192)
    tmp_4 = torch.roll(tmp_3, shifts=(3, 3), dims=(1, 2))
    tmp_5 = tmp_4[(slice(None, None, None), slice(None, 64, None), slice(None, 64, None), slice(None, None, None))]
    tmp_6 = tmp_5.contiguous()
    tmp_7 = tmp_6.view(1, 4096, 192)
    tmp_8 = in_2 + tmp_7
    tmp_9 = torch.nn.functional.layer_norm(tmp_8, (192,), in_1, in_0, 1e-05)
    return (tmp_8, tmp_9)


def replacement_args(in_0, in_1, in_2, in_3):
    return (in_0, in_1, in_2, in_3)


@triton.jit
def _fused_roll_add_norm_70_64_192(
    in2_ptr, in3_ptr, weight_ptr, bias_ptr,
    out_sum_ptr, out_norm_ptr,
    in3_s1, in3_s2, in3_s3, in3_s4, in3_s5,
    in2_s1, in2_s2,
    DTYPE: tl.constexpr,
    BLOCK_C: tl.constexpr,
    ROWS_PER_PROG: tl.constexpr,
):
    H: tl.constexpr = 70
    W: tl.constexpr = 70
    WP: tl.constexpr = 64
    C: tl.constexpr = 192
    EPS: tl.constexpr = 1e-5

    prog_id = tl.program_id(0)
    c_offsets = tl.arange(0, BLOCK_C)
    mask = c_offsets < C

    # Load weight and bias once per program (shared across ROWS_PER_PROG rows)
    w = tl.load(weight_ptr + c_offsets, mask=mask, other=1.0).to(tl.float32)
    b = tl.load(bias_ptr  + c_offsets, mask=mask, other=0.0).to(tl.float32)

    for r in tl.static_range(ROWS_PER_PROG):
        row_idx = prog_id * ROWS_PER_PROG + r
        i = row_idx // WP
        j = row_idx % WP

        # roll(shifts=(3,3)): output[i] = input[(i-3) % H]
        src_i = (i + H - 3) % H
        src_j = (j + W - 3) % W

        # in3 layout: [1, NH=10, 7, NW=10, 7, C=192]
        nh = src_i // 7
        h7 = src_i % 7
        nw = src_j // 7
        w7 = src_j % 7

        in3_base = nh * in3_s1 + h7 * in3_s2 + nw * in3_s3 + w7 * in3_s4
        in3_v = tl.load(in3_ptr + in3_base + c_offsets * in3_s5, mask=mask, other=0.0).to(tl.float32)

        in2_base = row_idx * in2_s1
        in2_v = tl.load(in2_ptr + in2_base + c_offsets * in2_s2, mask=mask, other=0.0).to(tl.float32)

        x = in2_v + in3_v  # residual add

        # Store out_sum in original dtype (matches baseline precision)
        x_stored = x.to(DTYPE)
        out_base = row_idx * C
        tl.store(out_sum_ptr + out_base + c_offsets, x_stored, mask=mask)

        # Layer norm: use dtype-rounded x to match baseline
        x_norm = x_stored.to(tl.float32)
        mean = tl.sum(x_norm, axis=0) / C
        diff = x_norm - mean
        # Zero masked positions so they don't corrupt the variance sum
        diff_sq = tl.where(mask, diff * diff, 0.0)
        var = tl.sum(diff_sq, axis=0) / C
        inv_std = tl.rsqrt(var + EPS)
        x_hat = diff * inv_std
        y = x_hat * w + b

        tl.store(out_norm_ptr + out_base + c_offsets, y.to(DTYPE), mask=mask)


@torch.fx.wrap
def _compute_70_64_192(in_0, in_1, in_2, in_3):
    N, C = 4096, 192
    ROWS_PER_PROG = 4
    dtype, device = in_2.dtype, in_2.device

    if dtype == torch.float16:
        td = tl.float16
    elif dtype == torch.bfloat16:
        td = tl.bfloat16
    else:
        td = tl.float32

    out_sum  = torch.empty(1, N, C, dtype=dtype, device=device)
    out_norm = torch.empty(1, N, C, dtype=dtype, device=device)

    grid = (N // ROWS_PER_PROG,)
    _fused_roll_add_norm_70_64_192[grid](
        in_2, in_3, in_1, in_0,
        out_sum, out_norm,
        in_3.stride(1), in_3.stride(2), in_3.stride(3), in_3.stride(4), in_3.stride(5),
        in_2.stride(1), in_2.stride(2),
        DTYPE=td, BLOCK_C=256, ROWS_PER_PROG=ROWS_PER_PROG,
        num_warps=8,
    )
    return out_sum, out_norm


def kernel_wrapper_70_64_192(in_0, in_1, in_2, in_3):
    result = _compute_70_64_192(in_0, in_1, in_2, in_3)
    return result[0], result[1]


def replacement_func():
    return kernel_wrapper_70_64_192