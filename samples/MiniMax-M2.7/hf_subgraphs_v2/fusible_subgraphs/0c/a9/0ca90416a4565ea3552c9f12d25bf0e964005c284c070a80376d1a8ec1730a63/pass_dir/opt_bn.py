import torch
import triton
import triton.language as tl

@triton.jit
def batch_norm_kernel(
    x_ptr, mean_ptr, var_ptr, weight_ptr, bias_ptr,
    out_ptr,
    BN_channels, H, W, eps: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid_c = tl.program_id(0)
    pid_bh = tl.program_id(1)
    
    batch_idx = pid_bh // H
    h_idx = pid_bh % H
    
    if pid_c >= BN_channels:
        return
    
    mean = tl.load(mean_ptr + pid_c)
    var = tl.load(var_ptr + pid_c)
    weight = tl.load(weight_ptr + pid_c)
    bias = tl.load(bias_ptr + pid_c)
    
    inv_std = tl.rsqrt(var + eps)
    scale = weight * inv_std
    shift = bias - weight * mean * inv_std
    
    offsets = pid_c * H * W + h_idx * W + tl.arange(0, BLOCK_SIZE)
    mask = h_idx * W + tl.arange(0, BLOCK_SIZE) < H * W
    
    x = tl.load(x_ptr + batch_idx * BN_channels * H * W + offsets, mask=mask, other=0.0)
    result = x * scale + shift
    tl.store(out_ptr + batch_idx * BN_channels * H * W + offsets, result, mask=mask)


@triton.jit
def slice_kernel(
    slice_ptr, out_ptr,
    slice_start, slice_total_channels,
    out_channels, H, W,
    BLOCK_SIZE: tl.constexpr,
):
    pid_c = tl.program_id(0)
    pid_bh = tl.program_id(1)
    
    batch_idx = pid_bh // H
    h_idx = pid_bh % H
    
    if pid_c >= out_channels:
        return
    
    slice_c_offset = pid_c + slice_start
    offsets = slice_c_offset * H * W + h_idx * W + tl.arange(0, BLOCK_SIZE)
    mask = h_idx * W + tl.arange(0, BLOCK_SIZE) < H * W
    
    data = tl.load(slice_ptr + batch_idx * slice_total_channels * H * W + offsets, mask=mask, other=0.0)
    tl.store(out_ptr + batch_idx * out_channels * H * W + offsets, data, mask=mask)


def pattern(in_0, in_1, in_2, in_3, in_4, in_5):
    """Match BN + getitem pattern for various slice values."""
    # Match the getitem pattern
    tmp_4 = in_5[(slice(None), slice(64, None), slice(None), slice(None))]
    # Match batch_norm
    tmp_5 = torch.nn.functional.batch_norm(in_4, in_0, in_1, in_3, in_2, False, 0.1, 0.001)
    return (tmp_5, tmp_4)


def replacement_args(in_0, in_1, in_2, in_3, in_4, in_5):
    return (in_0, in_1, in_2, in_3, in_4, in_5, 64, "route_64")


def pattern_128(in_0, in_1, in_2, in_3, in_4, in_5):
    """Match BN + getitem pattern for slice value 128."""
    tmp_4 = in_5[(slice(None), slice(128, None), slice(None), slice(None))]
    tmp_5 = torch.nn.functional.batch_norm(in_4, in_0, in_1, in_3, in_2, False, 0.1, 0.001)
    return (tmp_5, tmp_4)


def replacement_args_128(in_0, in_1, in_2, in_3, in_4, in_5):
    return (in_0, in_1, in_2, in_3, in_4, in_5, 128, "route_128")


def pattern_256(in_0, in_1, in_2, in_3, in_4, in_5):
    """Match BN + getitem pattern for slice value 256."""
    tmp_4 = in_5[(slice(None), slice(256, None), slice(None), slice(None))]
    tmp_5 = torch.nn.functional.batch_norm(in_4, in_0, in_1, in_3, in_2, False, 0.1, 0.001)
    return (tmp_5, tmp_4)


def replacement_args_256(in_0, in_1, in_2, in_3, in_4, in_5):
    return (in_0, in_1, in_2, in_3, in_4, in_5, 256, "route_256")


def pattern_512(in_0, in_1, in_2, in_3, in_4, in_5):
    """Match BN + getitem pattern for slice value 512."""
    tmp_4 = in_5[(slice(None), slice(512, None), slice(None), slice(None))]
    tmp_5 = torch.nn.functional.batch_norm(in_4, in_0, in_1, in_3, in_2, False, 0.1, 0.001)
    return (tmp_5, tmp_4)


def replacement_args_512(in_0, in_1, in_2, in_3, in_4, in_5):
    return (in_0, in_1, in_2, in_3, in_4, in_5, 512, "route_512")


def pattern_1024(in_0, in_1, in_2, in_3, in_4, in_5):
    """Match BN + getitem pattern for slice value 1024."""
    tmp_4 = in_5[(slice(None), slice(1024, None), slice(None), slice(None))]
    tmp_5 = torch.nn.functional.batch_norm(in_4, in_0, in_1, in_3, in_2, False, 0.1, 0.001)
    return (tmp_5, tmp_4)


def replacement_args_1024(in_0, in_1, in_2, in_3, in_4, in_5):
    return (in_0, in_1, in_2, in_3, in_4, in_5, 1024, "route_1024")


def pattern_2048(in_0, in_1, in_2, in_3, in_4, in_5):
    """Match BN + getitem pattern for slice value 2048."""
    tmp_4 = in_5[(slice(None), slice(2048, None), slice(None), slice(None))]
    tmp_5 = torch.nn.functional.batch_norm(in_4, in_0, in_1, in_3, in_2, False, 0.1, 0.001)
    return (tmp_5, tmp_4)


def replacement_args_2048(in_0, in_1, in_2, in_3, in_4, in_5):
    return (in_0, in_1, in_2, in_3, in_4, in_5, 2048, "route_2048")


@torch.fx.wrap
def bn_slice_dispatcher(in_0, in_1, in_2, in_3, in_4, in_5, slice_start, route):
    """Unified dispatcher for batch_norm and slice operations."""
    BN_channels = in_4.shape[1]
    H, W = in_4.shape[2], in_4.shape[3]
    B = in_4.shape[0]
    eps = 0.001
    slice_total_channels = in_5.shape[1]
    out_slice_channels = slice_total_channels - slice_start
    
    out_bn = torch.empty_like(in_4)
    out_slice = torch.empty(B, out_slice_channels, H, W, dtype=in_5.dtype, device=in_5.device)
    
    BLOCK_SIZE = 64
    grid_c = BN_channels
    grid_bh = B * H
    
    batch_norm_kernel[(grid_c, grid_bh)](
        x_ptr=in_4, mean_ptr=in_0, var_ptr=in_1, weight_ptr=in_3, bias_ptr=in_2,
        out_ptr=out_bn,
        BN_channels=BN_channels, H=H, W=W, eps=eps,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    grid_slice_c = out_slice_channels
    slice_kernel[(grid_slice_c, grid_bh)](
        slice_ptr=in_5, out_ptr=out_slice,
        slice_start=slice_start, slice_total_channels=slice_total_channels,
        out_channels=out_slice_channels, H=H, W=W,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return (out_bn, out_slice)


def replacement_func():
    """Returns the dispatcher function."""
    return bn_slice_dispatcher