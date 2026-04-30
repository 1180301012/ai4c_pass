import torch
import triton
import triton.language as tl


def pattern(tmp_5, in_6, tmp_8):
    """Match multiply -> pad -> add -> transpose(1,2) subgraph.
    
    This pattern matches across all target graphs because the pad dimensions
    (0,0,1,0,0,0) and transpose args (1,2) are fixed constants.
    tmp_5 is the transposed cat result, in_6 is q_img, tmp_8 is scale*in_4.
    """
    tmp_6 = in_6 * tmp_5
    tmp_7 = torch.nn.functional.pad(tmp_6, (0, 0, 1, 0, 0, 0), 'constant', None)
    tmp_9 = tmp_8 + tmp_7
    tmp_10 = tmp_9.transpose(1, 2)
    return (tmp_10,)


def replacement_args(tmp_5, in_6, tmp_8):
    return (tmp_5, in_6, tmp_8, "route_mul_pad_add_trans")


@triton.jit
def fused_mul_pad_add_trans_kernel(
    # Input pointers
    transposed_cat_ptr,
    in_6_ptr,
    tmp_8_ptr,
    output_ptr,
    # Dimensions
    seq_len,
    num_heads,
    head_dim,
    spatial_size,
    total_elements,
    # Strides for transposed_cat (may be non-contiguous from transpose(-1,-2))
    tc_stride_h,  # stride for num_heads dim = head_dim * spatial_size
    tc_stride_s,  # stride for spatial dim = 1
    tc_stride_d,  # stride for head_dim dim = spatial_size
    # Strides for in_6 (contiguous)
    in6_stride_h,
    in6_stride_s,
    in6_stride_d,
    # Strides for tmp_8 (contiguous)
    t8_stride_h,
    t8_stride_i,
    t8_stride_d,
    # Block size
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < total_elements
    
    # Decompose offset into (i, h, d) indices
    # output layout: [1, seq_len, num_heads, head_dim] (contiguous)
    # offset = i * (num_heads * head_dim) + h * head_dim + d
    nh_dim = num_heads * head_dim
    i = offsets // nh_dim
    remainder = offsets % nh_dim
    h = remainder // head_dim
    d = remainder % head_dim
    
    # Compute spatial_idx = i - 1 (for accessing in_6 and transposed_cat)
    spatial_idx = i - 1
    
    # Valid mask for spatial access (i >= 1 means spatial_idx >= 0)
    spatial_valid = (i >= 1) & mask
    
    # Read in_6[h, spatial_idx, d] 
    in6_offset = h * in6_stride_h + spatial_idx * in6_stride_s + d * in6_stride_d
    in6_val = tl.load(in_6_ptr + in6_offset, mask=spatial_valid, other=0.0)
    
    # Read transposed_cat[h, spatial_idx, d] (non-contiguous strides)
    tc_offset = h * tc_stride_h + spatial_idx * tc_stride_s + d * tc_stride_d
    tc_val = tl.load(transposed_cat_ptr + tc_offset, mask=spatial_valid, other=0.0)
    
    # Multiply: in_6 * transposed_cat
    mul_result = in6_val * tc_val
    
    # Pad: if i == 0, padded_result = 0; else padded_result = mul_result
    padded_result = tl.where(i >= 1, mul_result, 0.0)
    
    # Read tmp_8[h, i, d] (the scaled factor_att)
    t8_offset = h * t8_stride_h + i * t8_stride_i + d * t8_stride_d
    t8_val = tl.load(tmp_8_ptr + t8_offset, mask=mask, other=0.0)
    
    # Add: tmp_8 + padded_result
    result = t8_val + padded_result
    
    # Write output at offset (already in contiguous [1, seq_len, num_heads, head_dim] layout)
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
    """Fused kernel for multiply -> pad -> add -> transpose(1,2)."""
    # Determine dimensions from tensor shapes
    # tmp_8 shape: [1, num_heads, seq_len, head_dim]
    # in_6 shape: [1, num_heads, spatial_size, head_dim]
    # transposed_cat shape: [1, num_heads, spatial_size, head_dim]
    
    num_heads = tmp_8.shape[1]
    seq_len = tmp_8.shape[2]
    head_dim = tmp_8.shape[3]
    spatial_size = in_6.shape[2]
    
    total_elements = seq_len * num_heads * head_dim
    
    # Allocate output: [1, seq_len, num_heads, head_dim] (contiguous, in transposed layout)
    output = torch.empty((1, seq_len, num_heads, head_dim), 
                          dtype=tmp_8.dtype, device=tmp_8.device)
    
    # Compute strides
    # transposed_cat: result of transpose(-1,-2) on contiguous [1, num_heads, head_dim, spatial_size]
    # strides: (num_heads*head_dim*spatial_size, head_dim*spatial_size, 1, spatial_size)
    tc_stride_h = head_dim * transposed_cat.shape[2]  # This is head_dim * spatial_size for contiguous
    tc_stride_s = transposed_cat.stride(2)  # stride for spatial dim
    tc_stride_d = transposed_cat.stride(3)  # stride for head_dim dim
    
    # in_6: contiguous [1, num_heads, spatial_size, head_dim]
    in6_stride_h = in_6.stride(1)
    in6_stride_s = in_6.stride(2)
    in6_stride_d = in_6.stride(3)
    
    # tmp_8: contiguous [1, num_heads, seq_len, head_dim]
    t8_stride_h = tmp_8.stride(1)
    t8_stride_i = tmp_8.stride(2)
    t8_stride_d = tmp_8.stride(3)
    
    BLOCK_SIZE = 1024
    grid = ((total_elements + BLOCK_SIZE - 1) // BLOCK_SIZE,)
    
    fused_mul_pad_add_trans_kernel[grid](
        transposed_cat_ptr=transposed_cat,
        in_6_ptr=in_6,
        tmp_8_ptr=tmp_8,
        output_ptr=output,
        seq_len=seq_len,
        num_heads=num_heads,
        head_dim=head_dim,
        spatial_size=spatial_size,
        total_elements=total_elements,
        tc_stride_h=tc_stride_h,
        tc_stride_s=tc_stride_s,
        tc_stride_d=tc_stride_d,
        in6_stride_h=in6_stride_h,
        in6_stride_s=in6_stride_s,
        in6_stride_d=in6_stride_d,
        t8_stride_h=t8_stride_h,
        t8_stride_i=t8_stride_i,
        t8_stride_d=t8_stride_d,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return output


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


def replacement_func():
    return dispatch_wrapper