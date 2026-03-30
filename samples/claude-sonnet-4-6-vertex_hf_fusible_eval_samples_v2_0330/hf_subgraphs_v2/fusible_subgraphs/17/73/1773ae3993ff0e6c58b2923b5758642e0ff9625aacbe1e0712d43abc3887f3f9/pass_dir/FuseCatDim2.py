import torch
import triton
import triton.language as tl


@triton.jit
def _cat_dim2_3way_kernel(
    a_ptr, b_ptr, c_ptr, out_ptr,
    B, S1, S2, S3, S_total, H,
    BLOCK_H: tl.constexpr,
):
    """
    Each program covers one (b_idx, s_idx) row of output, processing BLOCK_H elements.
    Grid: (B, S_total, ceil(H / BLOCK_H))
    """
    b_idx   = tl.program_id(0)
    s_idx   = tl.program_id(1)
    h_block = tl.program_id(2)

    h_off  = h_block * BLOCK_H + tl.arange(0, BLOCK_H)
    h_mask = h_off < H

    out_off = b_idx * S_total * H + s_idx * H

    use_a = s_idx < S1
    use_b = (s_idx >= S1) & (s_idx < S1 + S2)

    s_a = s_idx
    s_b = tl.where(s_idx >= S1, s_idx - S1, 0)
    s_c = tl.where(s_idx >= S1 + S2, s_idx - S1 - S2, 0)

    data_a = tl.load(a_ptr + b_idx * S1 * H + s_a * H + h_off,
                     mask=h_mask & use_a, other=0.0)
    data_b = tl.load(b_ptr + b_idx * S2 * H + s_b * H + h_off,
                     mask=h_mask & use_b, other=0.0)
    data_c = tl.load(c_ptr + b_idx * S3 * H + s_c * H + h_off,
                     mask=h_mask & (~use_a & ~use_b), other=0.0)

    data = data_a + data_b + data_c

    tl.store(out_ptr + out_off + h_off, data, mask=h_mask)


@torch.fx.wrap
def triton_cat_dim2(a, b, c):
    """
    Concatenate three tensors along dim=2.
    Inputs: [B, 1, S_i, H]. Output: [B, 1, S1+S2+S3, H].
    """
    B  = a.shape[0]
    S1 = a.shape[2]
    S2 = b.shape[2]
    S3 = c.shape[2]
    H  = a.shape[3]
    S_total = S1 + S2 + S3

    # Choose BLOCK_H: 32 for small H (avoids 16x waste), 512 for large H
    if H <= 32:
        BLOCK_H = 32
    else:
        BLOCK_H = 512

    H_blocks = (H + BLOCK_H - 1) // BLOCK_H

    a_flat = a.reshape(B, S1, H)
    b_flat = b.reshape(B, S2, H)
    c_flat = c.reshape(B, S3, H)
    out_flat = a_flat.new_empty(B, S_total, H)

    grid = (B, S_total, H_blocks)

    _cat_dim2_3way_kernel[grid](
        a_flat, b_flat, c_flat, out_flat,
        B, S1, S2, S3, S_total, H,
        BLOCK_H=BLOCK_H,
    )

    return out_flat.reshape(B, 1, S_total, H)


def pattern(a, b, c):
    return torch.cat((a, b, c), dim=2)


def replacement_args(a, b, c):
    return (a, b, c)


def replacement_func():
    return triton_cat_dim2