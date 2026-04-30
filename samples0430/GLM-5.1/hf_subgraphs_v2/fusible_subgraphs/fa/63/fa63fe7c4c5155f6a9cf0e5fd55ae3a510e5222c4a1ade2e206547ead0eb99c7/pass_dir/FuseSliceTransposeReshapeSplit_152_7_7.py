import torch
import triton
import triton.language as tl


@triton.jit
def fuse_slice_transpose_reshape_split_kernel(
    in_ptr,
    out0_ptr,
    out1_ptr,
    out2_ptr,
    seq_len,
    head_dim: tl.constexpr,
    H: tl.constexpr,
    W: tl.constexpr,
    total_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < total_elements

    HW = H * W
    group0_end = 2 * head_dim
    group1_end = 5 * head_dim

    # Decode flat index: offsets = c * HW + spatial_idx
    spatial_idx = offsets % HW
    c = offsets // HW

    # Decode c to (head_idx, dim_idx)
    head_idx = c // head_dim
    dim_idx = c % head_dim

    # Compute input sequence index (skip first token: offset by 1)
    seq_idx = spatial_idx + 1

    # Input offset: in_2[0, head_idx, seq_idx, dim_idx]
    # in_2 layout: [1, 8, seq_len, head_dim]
    in_offset = head_idx * seq_len * head_dim + seq_idx * head_dim + dim_idx

    # Load from input
    val = tl.load(in_ptr + in_offset, mask=mask, other=0.0)

    # Determine output group and compute write offsets
    mask0 = c < group0_end
    mask1 = (c >= group0_end) & (c < group1_end)
    mask2 = c >= group1_end

    # out0: [1, group0_ch, H, W], out1: [1, group1_ch, H, W], out2: [1, group2_ch, H, W]
    tl.store(out0_ptr + c * HW + spatial_idx, val, mask=mask & mask0)
    tl.store(out1_ptr + (c - group0_end) * HW + spatial_idx, val, mask=mask & mask1)
    tl.store(out2_ptr + (c - group1_end) * HW + spatial_idx, val, mask=mask & mask2)


def _fuse_impl(in_2, head_dim, H, W):
    seq_len = in_2.shape[2]
    group0_ch = 2 * head_dim
    group1_ch = 3 * head_dim
    group2_ch = 3 * head_dim
    total_elements = 8 * head_dim * H * W

    BLOCK_SIZE = 1024
    if total_elements < BLOCK_SIZE:
        BLOCK_SIZE = max(triton.next_power_of_2(total_elements), 32)

    num_programs = (total_elements + BLOCK_SIZE - 1) // BLOCK_SIZE

    out0 = torch.empty((1, group0_ch, H, W), dtype=in_2.dtype, device=in_2.device)
    out1 = torch.empty((1, group1_ch, H, W), dtype=in_2.dtype, device=in_2.device)
    out2 = torch.empty((1, group2_ch, H, W), dtype=in_2.dtype, device=in_2.device)

    fuse_slice_transpose_reshape_split_kernel[(num_programs,)](
        in_ptr=in_2,
        out0_ptr=out0,
        out1_ptr=out1,
        out2_ptr=out2,
        seq_len=seq_len,
        head_dim=head_dim,
        H=H,
        W=W,
        total_elements=total_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    return out0, out1, out2


@torch.fx.wrap
def dispatch_fuse(in_2, route):
    if route == "r19_7_7":
        return _fuse_impl(in_2, head_dim=19, H=7, W=7)
    elif route == "r19_14_14":
        return _fuse_impl(in_2, head_dim=19, H=14, W=14)
    elif route == "r19_28_28":
        return _fuse_impl(in_2, head_dim=19, H=28, W=28)
    elif route == "r19_56_56":
        return _fuse_impl(in_2, head_dim=19, H=56, W=56)
    elif route == "r40_7_7":
        return _fuse_impl(in_2, head_dim=40, H=7, W=7)
    elif route == "r40_14_14":
        return _fuse_impl(in_2, head_dim=40, H=14, W=14)
    elif route == "r27_7_7":
        return _fuse_impl(in_2, head_dim=27, H=7, W=7)
    elif route == "r27_14_14":
        return _fuse_impl(in_2, head_dim=27, H=14, W=14)
    elif route == "r27_28_28":
        return _fuse_impl(in_2, head_dim=27, H=28, W=28)
    elif route == "r32_14_14":
        return _fuse_impl(in_2, head_dim=32, H=14, W=14)
    elif route == "r32_48_48":
        return _fuse_impl(in_2, head_dim=32, H=48, W=48)
    elif route == "r16_28_28":
        return _fuse_impl(in_2, head_dim=16, H=28, W=28)
    elif route == "r64_12_12":
        return _fuse_impl(in_2, head_dim=64, H=12, W=12)
    else:
        raise ValueError(f"Unknown route: {route}")


# Pattern for (1, 152, 7, 7) reshape with split [38, 57, 57]
def pattern(in_2):
    tmp_2 = in_2[slice(None, None, None), slice(None, None, None), slice(1, None, None), slice(None, None, None)]
    tmp_3 = tmp_2.transpose(-1, -2)
    tmp_4 = tmp_3.reshape(1, 152, 7, 7)
    tmp_5 = torch.functional.split(tmp_4, [38, 57, 57], dim=1)
    tmp_6 = tmp_5[0]
    tmp_7 = tmp_5[1]
    tmp_8 = tmp_5[2]
    return (tmp_6, tmp_7, tmp_8)


def replacement_args(in_2):
    return (in_2, "r19_7_7")


def replacement_func():
    return dispatch_fuse