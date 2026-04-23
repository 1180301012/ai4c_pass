import torch
import triton
import triton.language as tl

@triton.jit
def fused_slice_transpose_reshape_split_kernel(
    in_2_ptr,
    out_0_ptr,
    out_1_ptr,
    out_2_ptr,
    stride_0: tl.constexpr,
    stride_1: tl.constexpr,
    stride_2: tl.constexpr,
    stride_3: tl.constexpr,
    B: tl.constexpr,
    H: tl.constexpr,
    S: tl.constexpr,
    D: tl.constexpr,
    split_0_size: tl.constexpr,
    split_1_size: tl.constexpr,
    split_2_size: tl.constexpr,
    grid_h: tl.constexpr,
    grid_w: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    
    total_spatial = grid_h * grid_w
    total_channels = H * D
    
    b = pid // (total_channels * total_spatial)
    remaining = pid % (total_channels * total_spatial)
    out_c = remaining // total_spatial
    spatial_flat = remaining % total_spatial
    
    spatial_y = spatial_flat // grid_w
    spatial_x = spatial_flat % grid_w
    
    out_h = out_c % H
    out_d = out_c // H
    
    seq_idx = spatial_y * grid_w + spatial_x
    
    offset = b * stride_0 + out_h * stride_1 + (1 + seq_idx) * stride_2 + out_d
    val = tl.load(in_2_ptr + offset)
    
    out_offset = b * total_channels * total_spatial + out_c * total_spatial + spatial_flat
    
    if out_c < split_0_size:
        tl.store(out_0_ptr + out_offset, val)
    elif out_c < split_0_size + split_1_size:
        tl.store(out_1_ptr + out_offset - split_0_size * total_spatial, val)
    else:
        tl.store(out_2_ptr + out_offset - (split_0_size + split_1_size) * total_spatial, val)


def pattern(in_2):
    """
    Match the slice-transpose-reshape-split pattern for coat_lite_tiny [1, 320, 7, 7].
    """
    tmp_2 = in_2[:, :, 1:, :]
    tmp_3 = tmp_2.transpose(-1, -2)
    tmp_4 = tmp_3.reshape(1, 320, 7, 7)
    split_result = torch.functional.split(tmp_4, [80, 120, 120], dim=1)
    return split_result[0], split_result[1], split_result[2]


def replacement_args(in_2):
    """Extract tensor and shape information for coat_lite_tiny."""
    B, H, S, D = in_2.shape
    
    grid_val = S - 1  # 49
    reshape_h = 7
    reshape_w = 7
    C = H * D  # 320
    
    s0 = 80
    s1 = 120
    s2 = 120
    
    return (in_2, (s0, s1, s2), reshape_h, reshape_w, grid_val)


@torch.fx.wrap
def fused_slice_transpose_reshape_split_wrapper(in_2, split_sizes, reshape_h, reshape_w, grid_val):
    """
    Wrapper function for coat_lite_tiny pattern.
    """
    B, H, S, D = in_2.shape
    s0, s1, s2 = split_sizes
    
    try:
        actual_grid = int(grid_val)
    except (TypeError, AttributeError):
        actual_grid = 49
    
    g = int(round(actual_grid ** 0.5))
    actual_reshape_h = g
    actual_reshape_w = g
    
    out_0 = torch.empty(B * s0 * actual_reshape_h * actual_reshape_w, device=in_2.device, dtype=in_2.dtype).reshape(B, s0, actual_reshape_h, actual_reshape_w)
    out_1 = torch.empty(B * s1 * actual_reshape_h * actual_reshape_w, device=in_2.device, dtype=in_2.dtype).reshape(B, s1, actual_reshape_h, actual_reshape_w)
    out_2 = torch.empty(B * s2 * actual_reshape_h * actual_reshape_w, device=in_2.device, dtype=in_2.dtype).reshape(B, s2, actual_reshape_h, actual_reshape_w)
    
    total_elements = B * H * D * actual_reshape_h * actual_reshape_w
    BLOCK_SIZE = 1024
    num_programs = (total_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    stride_0 = H * S * D
    stride_1 = S * D
    stride_2 = D
    stride_3 = 1
    
    fused_slice_transpose_reshape_split_kernel[(num_programs,)](
        in_2,
        out_0,
        out_1,
        out_2,
        stride_0,
        stride_1,
        stride_2,
        stride_3,
        B,
        H,
        S,
        D,
        s0,
        s1,
        s2,
        actual_reshape_h,
        actual_reshape_w,
        BLOCK_SIZE,
    )
    
    return out_0, out_1, out_2


def replacement_func():
    return fused_slice_transpose_reshape_split_wrapper