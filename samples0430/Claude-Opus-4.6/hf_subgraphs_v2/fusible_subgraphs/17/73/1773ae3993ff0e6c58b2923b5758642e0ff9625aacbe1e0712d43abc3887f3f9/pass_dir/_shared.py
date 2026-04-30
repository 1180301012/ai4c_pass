import torch
import triton
import triton.language as tl


@triton.jit
def _ln_kernel(
    X, W, B, Y,
    N_rows,
    D: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    ROWS_PER_PROG: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = tl.arange(0, BLOCK_SIZE)
    mask = offs < D

    # Load weight and bias once (reused for all rows in this program)
    w = tl.load(W + offs, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(B + offs, mask=mask, other=0.0).to(tl.float32)

    for i in range(ROWS_PER_PROG):
        row = pid * ROWS_PER_PROG + i
        row_mask = row < N_rows
        combined_mask = mask & row_mask

        base = row * D
        x = tl.load(X + base + offs, mask=combined_mask, other=0.0).to(tl.float32)

        # Single-pass mean and variance: var = E[x^2] - E[x]^2
        sum_x = tl.sum(x, axis=0)
        sum_x2 = tl.sum(x * x, axis=0)
        mean = sum_x / D
        var = sum_x2 / D - mean * mean
        rstd = tl.math.rsqrt(var + 1e-12)

        # Normalize, scale and shift
        out = (x - mean) * rstd * w + b
        tl.store(Y + base + offs, out, mask=combined_mask)


@triton.jit
def _cat3_dim2_kernel(
    src1, src2, src3, dst,
    N1D, N12D, N_total_D,
    src1_batch_stride, src2_batch_stride, src3_batch_stride,
    BLOCK_SIZE: tl.constexpr,
):
    batch = tl.program_id(0)
    seg = tl.program_id(1)

    offs = seg * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < N_total_D

    # Determine source
    is_src1 = offs < N1D
    is_src2 = (offs >= N1D) & (offs < N12D)

    # Compute source positions
    src1_pos = batch * src1_batch_stride + offs
    src2_pos = batch * src2_batch_stride + (offs - N1D)
    src3_pos = batch * src3_batch_stride + (offs - N12D)

    # Conditional load
    val = tl.where(is_src1,
                   tl.load(src1 + src1_pos, mask=mask & is_src1, other=0.0),
                   tl.where(is_src2,
                            tl.load(src2 + src2_pos, mask=mask & is_src2, other=0.0),
                            tl.load(src3 + src3_pos, mask=mask & (~is_src1 & ~is_src2), other=0.0)))

    # Store to output
    dst_pos = batch * N_total_D + offs
    tl.store(dst + dst_pos, val, mask=mask)


@torch.fx.wrap
def triton_ln_dispatch(x, weight, bias, route):
    out = torch.empty_like(x)
    D = x.shape[-1]
    N = x.numel() // D
    BLOCK_SIZE = triton.next_power_of_2(D)

    if D <= 64:
        ROWS_PER_PROG = 8
        num_warps = 2
    elif D <= 512:
        ROWS_PER_PROG = 4
        num_warps = 2
    else:
        ROWS_PER_PROG = 2
        num_warps = 8

    num_programs = (N + ROWS_PER_PROG - 1) // ROWS_PER_PROG
    _ln_kernel[(num_programs,)](
        x, weight, bias, out,
        N,
        D=D, BLOCK_SIZE=BLOCK_SIZE, ROWS_PER_PROG=ROWS_PER_PROG,
        num_warps=num_warps,
    )
    return out


@torch.fx.wrap
def triton_full_dispatch(in_0, in_1, in_2, in_3, in_4, in_5, route):
    # Layer norm on in_4
    D = in_4.shape[-1]
    N_ln = in_4.numel() // D
    ln_out = torch.empty_like(in_4)
    BLOCK_SIZE_LN = triton.next_power_of_2(D)

    if D <= 64:
        ROWS_PER_PROG = 8
        num_warps_ln = 2
    elif D <= 512:
        ROWS_PER_PROG = 4
        num_warps_ln = 4
    else:
        ROWS_PER_PROG = 2
        num_warps_ln = 8

    num_programs_ln = (N_ln + ROWS_PER_PROG - 1) // ROWS_PER_PROG
    _ln_kernel[(num_programs_ln,)](
        in_4, in_1, in_0, ln_out,
        N_ln,
        D=D, BLOCK_SIZE=BLOCK_SIZE_LN, ROWS_PER_PROG=ROWS_PER_PROG,
        num_warps=num_warps_ln,
    )

    # Cat: (in_2, in_5, in_3) along dim=2
    B = in_2.shape[0]
    N1 = in_2.shape[2]
    N2 = in_5.shape[2]
    N3 = in_3.shape[2]
    D_cat = in_2.shape[3]
    N_total = N1 + N2 + N3

    cat_out = torch.empty(B, 1, N_total, D_cat, dtype=in_2.dtype, device=in_2.device)

    N1D = N1 * D_cat
    N2D = N2 * D_cat
    N3D = N3 * D_cat
    N12D = N1D + N2D
    N_total_D = N1D + N2D + N3D

    BLOCK_SIZE_CAT = 1024
    grid_cat = (B, (N_total_D + BLOCK_SIZE_CAT - 1) // BLOCK_SIZE_CAT)
    _cat3_dim2_kernel[grid_cat](
        in_2, in_5, in_3, cat_out,
        N1D, N12D, N_total_D,
        N1D, N2D, N3D,
        BLOCK_SIZE=BLOCK_SIZE_CAT,
        num_warps=4,
    )

    return (ln_out, cat_out)