import torch
import triton
import triton.language as tl

# N3+N4 = 6400+1600 = 8000  (in3+in4 section size)
# N_A = N3+N4+N5 = 8000+400 = 8400 (total output elements per batch-channel)
# N5 = 400 (ic/conv section size)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 256}),
        triton.Config({'BLOCK_SIZE': 512}),
        triton.Config({'BLOCK_SIZE': 1024}),
    ],
    key=['N_total'],
)
@triton.jit
def fused_cat_sigmoid_sub_mul_kernel(
    in_3_ptr,
    in_4_ptr,
    conv_ptr,
    out_ptr,
    B,
    N_total,
    N_A,                   # N3 + N4  (boundary between in3/in4 and conv section)
    N5,                    # conv / ic output elements per batch
    BLOCK_SIZE: tl.constexpr,
):
    # 2-D grid: dim-0 = batch, dim-1 = tile within output row
    b    = tl.program_id(0)
    tile = tl.program_id(1)
    offs = tile * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)

    mask = offs < N_total

    # Pos index in the concatenated output row (0..N_total-1).
    # For offs >= N_total (padding beyond the last valid element):
    #   `invalid=True` means the masked store zeroes the write, but we still
    #   need a finite pointer — clamp to 0, which is never read.
    invalid = offs >= N_total
    n = tl.where(mask, offs, 0)

    # in3+in4 section: n in [0, N_A)
    # conv/IC section: n in [N_A,  N_A+N5)
    in_ic = n >= N_A
    # Clamped ic index so that out-of-the-in3/in4 range never tries to index
    # beyond conv's [0, N5) bounds.  (For valid ic elements n_ic is always in
    # [0, N5), so tl.maximum(n_ic - N5, 0) == 0 there and is safe.)
    n_ic = tl.maximum(n - N_A, 0)

    # Conjunction of "in-range" and "not in the padding suffix": same as mask
    # but structuralises the region description.
    in_first = mask & ~invalid          # elements genuinely in [0, N_total)
    in_ic_f  = in_ic & in_first        # and belong to the IC section

    # p3/p4: src = in_3 or in_4, active only for first section
    is_in3 = in_first & (n < N_A)
    is_in4 = in_ic_f  & (n < N_A + N5)

    x_p3 = tl.load(in_3_ptr + (b * N_A + n), mask=is_in3, other=0.0)
    x_p4 = tl.load(in_4_ptr + (b * N_A + n - N_A), mask=is_in4, other=0.0)
    x_p5 = tl.load(conv_ptr + (b * N5 + tl.maximum(n_ic - N5, 0)),
                   mask=in_ic_f, other=0.0)

    # Exactly one of x_p3, x_p4, x_p5 is non-zero per element.
    x = x_p3 + x_p4 + x_p5

    # Numerically stable sigmoid in fp32, then back to the original input type
    x_f32 = x.to(tl.float32)
    y_f32  = 1.0 / (1.0 + tl.exp(-x_f32))
    result = (y_f32 - 0.25) * 3.141592653589793

    out = result.to(x.dtype)
    tl.store(out_ptr + (b * N_total + n), out, mask=mask)


@torch.fx.wrap
def fused_cat_sigmoid_sub_mul(in_3, in_4, conv_out):
    """
    Fused replacement for:
        cat([in_3, in_4, conv_out], dim=2)
        .sigmoid()
        - 0.25
        * 3.141592653589793

    in_3:     [B, 1, N3]  (e.g. [B,1,6400])
    in_4:     [B, 1, N4]  (e.g. [B,1,1600])
    conv_out: [B, 1, N5]  (e.g. [B,1, 400])
    returns:  [B, 1, N3+N4+N5]  = [B,1,8400]
    """
    B      = in_3.shape[0]
    N_A    = in_3.shape[2] + in_4.shape[2]   # 6400 + 1600 = 8000
    N5     = conv_out.shape[2]                 # 400
    N_total = N_A + N5                         # 8400

    out = torch.empty((B, 1, N_total), dtype=in_3.dtype, device=in_3.device)

    # 2-D grid: (batch, tiles)
    grid = lambda meta: (B, (N_total + meta['BLOCK_SIZE'] - 1) // meta['BLOCK_SIZE'])

    fused_cat_sigmoid_sub_mul_kernel[grid](
        in_3, in_4, conv_out, out,
        B, N_total, N_A, N5,
    )
    return out


# ---------------------------------------------------------------------------
# Pattern, replacement_args, replacement_func
# ---------------------------------------------------------------------------

def pattern(in_3, in_4, tmp_3):
    tmp_4 = torch.cat([in_3, in_4, tmp_3], 2)
    tmp_5 = tmp_4.sigmoid()
    tmp_6 = tmp_5 - 0.25
    tmp_7 = tmp_6 * 3.141592653589793
    return tmp_7


def replacement_args(in_3, in_4, tmp_3):
    return (in_3, in_4, tmp_3)


def replacement_func():
    return fused_cat_sigmoid_sub_mul