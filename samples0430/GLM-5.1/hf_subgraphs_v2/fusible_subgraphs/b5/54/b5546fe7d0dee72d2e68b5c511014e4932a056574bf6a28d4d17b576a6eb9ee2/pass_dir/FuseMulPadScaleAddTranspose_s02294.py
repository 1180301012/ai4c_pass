import torch
import triton
import triton.language as tl


def pattern(tmp_5, in_6, in_4):
    """Match multiply -> pad -> scale -> add -> transpose(1,2) with scale=0.22941573387056177."""
    tmp_6 = in_6 * tmp_5
    tmp_7 = torch.nn.functional.pad(tmp_6, (0, 0, 1, 0, 0, 0), 'constant', None)
    tmp_8 = 0.22941573387056177 * in_4
    tmp_9 = tmp_8 + tmp_7
    tmp_10 = tmp_9.transpose(1, 2)
    return (tmp_10,)


def replacement_args(tmp_5, in_6, in_4):
    return (tmp_5, in_6, in_4, "route_mul_pad_scale_add_trans_02294")


@triton.jit
def fused_mul_pad_scale_add_trans_kernel(
    transposed_cat_ptr,
    in_6_ptr,
    in_4_ptr,
    output_ptr,
    seq_len,
    num_heads,
    head_dim,
    spatial_size,
    total_elements,
    scale,
    tc_stride_h,
    tc_stride_s,
    tc_stride_d,
    in6_stride_h,
    in6_stride_s,
    in6_stride_d,
    in4_stride_h,
    in4_stride_i,
    in4_stride_d,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < total_elements
    
    nh_dim = num_heads * head_dim
    i = offsets // nh_dim
    remainder = offsets % nh_dim
    h = remainder // head_dim
    d = remainder % head_dim
    
    spatial_idx = i - 1
    spatial_valid = (i >= 1) & mask
    
    # Read in_6
    in6_offset = h * in6_stride_h + spatial_idx * in6_stride_s + d * in6_stride_d
    in6_val = tl.load(in_6_ptr + in6_offset, mask=spatial_valid, other=0.0)
    
    # Read transposed_cat
    tc_offset = h * tc_stride_h + spatial_idx * tc_stride_s + d * tc_stride_d
    tc_val = tl.load(transposed_cat_ptr + tc_offset, mask=spatial_valid, other=0.0)
    
    # Multiply
    mul_result = in6_val * tc_val
    
    # Pad
    padded_result = tl.where(i >= 1, mul_result, 0.0)
    
    # Read in_4 and scale
    in4_offset = h * in4_stride_h + i * in4_stride_i + d * in4_stride_d
    in4_val = tl.load(in_4_ptr + in4_offset, mask=mask, other=0.0)
    scaled_val = scale * in4_val
    
    # Add
    result = scaled_val + padded_result
    
    # Write output
    tl.store(output_ptr + offsets, result, mask=mask)


@torch.fx.wrap
def dispatch_wrapper(transposed_cat, in_6, tmp_8_or_in4, route):
    """Shared dispatch wrapper for all pass routes."""
    if route == "route_mul_pad_add_trans":
        return _fuse_mul_pad_add_trans(transposed_cat, in_6, tmp_8_or_in4)
    elif route == "route_mul_pad_scale_add_trans_02294":
        return _fuse_mul_pad_scale_add_trans(transposed_cat, in_6, tmp_8_or_in4, 0.22941573387056177)
    elif route == "route_mul_pad_scale_add_trans_01924":
        return _fuse_mul_pad_scale_add_trans(transposed_cat, in_6, tmp_8_or_in4, 0.19245008972987526)
    elif route == "route_mul_pad_scale_add_trans_01767":
        return _fuse_mul_pad_scale_add_trans(transposed_cat, in_6, tmp_8_or_in4, 0.1767766952966369)
    elif route == "route_mul_pad_scale_add_trans_01581":
        return _fuse_mul_pad_scale_add_trans(transposed_cat, in_6, tmp_8_or_in4, 0.15811388300841897)
    elif route == "route_mul_pad_scale_add_trans_0125":
        return _fuse_mul_pad_scale_add_trans(transposed_cat, in_6, tmp_8_or_in4, 0.125)
    else:
        raise ValueError(f"Unknown route: {route}")


def _fuse_mul_pad_add_trans(transposed_cat, in_6, tmp_8):
    """Placeholder for route_mul_pad_add_trans - never called in this pass's context."""
    raise NotImplementedError("This route is not used in this pass")


def _fuse_mul_pad_scale_add_trans(transposed_cat, in_6, in_4, scale):
    """Fused kernel for multiply -> pad -> scale -> add -> transpose(1,2)."""
    num_heads = in_4.shape[1]
    seq_len = in_4.shape[2]
    head_dim = in_4.shape[3]
    spatial_size = in_6.shape[2]
    
    total_elements = seq_len * num_heads * head_dim
    
    output = torch.empty((1, seq_len, num_heads, head_dim),
                          dtype=in_4.dtype, device=in_4.device)
    
    # Compute strides
    tc_stride_h = transposed_cat.stride(1)
    tc_stride_s = transposed_cat.stride(2)
    tc_stride_d = transposed_cat.stride(3)
    
    in6_stride_h = in_6.stride(1)
    in6_stride_s = in_6.stride(2)
    in6_stride_d = in_6.stride(3)
    
    in4_stride_h = in_4.stride(1)
    in4_stride_i = in_4.stride(2)
    in4_stride_d = in_4.stride(3)
    
    BLOCK_SIZE = 1024
    grid = ((total_elements + BLOCK_SIZE - 1) // BLOCK_SIZE,)
    
    fused_mul_pad_scale_add_trans_kernel[grid](
        transposed_cat_ptr=transposed_cat,
        in_6_ptr=in_6,
        in_4_ptr=in_4,
        output_ptr=output,
        seq_len=seq_len,
        num_heads=num_heads,
        head_dim=head_dim,
        spatial_size=spatial_size,
        total_elements=total_elements,
        scale=scale,
        tc_stride_h=tc_stride_h,
        tc_stride_s=tc_stride_s,
        tc_stride_d=tc_stride_d,
        in6_stride_h=in6_stride_h,
        in6_stride_s=in6_stride_s,
        in6_stride_d=in6_stride_d,
        in4_stride_h=in4_stride_h,
        in4_stride_i=in4_stride_i,
        in4_stride_d=in4_stride_d,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return output


def replacement_func():
    return dispatch_wrapper