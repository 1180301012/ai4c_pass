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
def fused_roll_slice_add_ln_kernel_192(
    in_3_ptr,
    in_2_ptr,
    in_1_ptr,
    in_0_ptr,
    out_8_ptr,
    out_9_ptr,
    N: tl.constexpr,
    H: tl.constexpr,
    W: tl.constexpr,
    crop_h: tl.constexpr,
    crop_w: tl.constexpr,
    C: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    row_idx = tl.program_id(0)

    # 2D position from flattened index
    row_2d = row_idx // crop_w
    col_2d = row_idx % crop_w

    # Reverse roll: source position in pre-roll tensor
    src_row = (row_2d - 3 + H) % H
    src_col = (col_2d - 3 + W) % W

    # Flat offset in in_3 (viewed as [1, H, W, C])
    in_3_row_offset = (src_row * W + src_col) * C

    c_offsets = tl.arange(0, BLOCK_C)
    mask = c_offsets < C

    # Load from in_3
    in_3_vals = tl.load(in_3_ptr + in_3_row_offset + c_offsets, mask=mask, other=0.0)

    # Load from in_2
    in_2_row_offset = row_idx * C
    in_2_vals = tl.load(in_2_ptr + in_2_row_offset + c_offsets, mask=mask, other=0.0)

    # Add: tmp_8 = in_2 + rolled_sliced_in_3
    tmp_8 = in_2_vals + in_3_vals

    # Store tmp_8
    tl.store(out_8_ptr + in_2_row_offset + c_offsets, tmp_8, mask=mask)

    # Layer norm computation in float32
    tmp_8_f32 = tmp_8.to(tl.float32)
    mean = tl.sum(tmp_8_f32, axis=0) / C
    diff = tmp_8_f32 - mean
    diff = tl.where(mask, diff, 0.0)
    var = tl.sum(diff * diff, axis=0) / C
    inv_std = 1.0 / tl.sqrt(var + 1e-5)
    normalized = diff * inv_std

    # Load weight and bias
    weight = tl.load(in_1_ptr + c_offsets, mask=mask, other=1.0).to(tl.float32)
    bias = tl.load(in_0_ptr + c_offsets, mask=mask, other=0.0).to(tl.float32)

    # Apply affine transform
    tmp_9 = normalized * weight + bias
    tmp_9 = tmp_9.to(tmp_8.dtype)

    # Store tmp_9
    tl.store(out_9_ptr + in_2_row_offset + c_offsets, tmp_9, mask=mask)


@torch.fx.wrap
def fused_roll_slice_add_ln_192(in_0, in_1, in_2, in_3):
    # Constants for C=192 variant
    H = 70
    W = 70
    crop_h = 64
    crop_w = 64
    C = 192
    N = crop_h * crop_w  # 4096
    BLOCK_C = 256

    # Make in_3 contiguous for correct memory layout
    in_3_contig = in_3.contiguous()

    out_8 = torch.empty_like(in_2)
    out_9 = torch.empty_like(in_2)

    grid = (N,)
    fused_roll_slice_add_ln_kernel_192[grid](
        in_3_contig, in_2, in_1, in_0,
        out_8, out_9,
        N, H, W, crop_h, crop_w, C, BLOCK_C,
    )

    return (out_8, out_9)


def replacement_func():
    return fused_roll_slice_add_ln_192