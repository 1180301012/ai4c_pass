import torch
import triton
import triton.language as tl


@triton.jit
def fused_post_conv_kernel(
    # Input pointers
    conv_ptr,       # conv2d output: [1, C_conv, H, W]
    in2_ptr,        # in_2: [1, C2, H, W]
    in3_ptr,        # in_3: [1, C3, H, W]
    in6_ptr,        # q_img: [1, heads, HW, C_inner]
    in4_ptr,        # factor_att: [1, heads, HW+1, C_inner]
    scale_ptr,      # scalar scale factor
    # Output pointer
    out_ptr,        # output: [1, HW+1, C_total] where C_total = heads * C_inner
    # Dimensions
    C2, C3, C_conv,
    H, W,
    heads, C_inner,
    HW,  # H * W
    C_total,  # heads * C_inner = (C2 + C3 + C_conv)
    # Conv2d output stride info (contiguous NCHW)
    conv_stride_c,
    conv_stride_h,
    conv_stride_w,
    # in_2 stride info
    in2_stride_c,
    in2_stride_h,
    in2_stride_w,
    # in_3 stride info
    in3_stride_c,
    in3_stride_h,
    in3_stride_w,
    # q_img strides [1, heads, HW, C_inner]
    in6_stride_h1,   # stride for heads dim
    in6_stride_hw,   # stride for HW dim
    in6_stride_ci,   # stride for C_inner dim
    # factor_att strides [1, heads, HW+1, C_inner]
    in4_stride_h1,
    in4_stride_seq,   # stride for seq (HW+1) dim
    in4_stride_ci,
    # Output strides [1, seq_len, C_total]
    out_stride_seq,
    out_stride_ct,
    # Block sizes
    BLOCK_CI: tl.constexpr,
):
    # Each program handles one (head, hw_block) pair for the HW rows
    pid = tl.program_id(0)
    
    # We have heads * HW total items to process
    # pid maps to (head_idx, hw_idx)
    head_idx = pid // HW
    hw_idx = pid % HW
    
    # Compute the spatial position (h, w) from hw_idx
    h_pos = hw_idx // W
    w_pos = hw_idx % W
    
    # Offsets for C_inner dimension
    ci_offsets = tl.arange(0, BLOCK_CI)
    ci_mask = ci_offsets < C_inner
    
    # Global channel index within the cat result
    global_ci = head_idx * C_inner + ci_offsets
    
    # Read from appropriate tensor based on global_ci
    in2_val = tl.load(in2_ptr + global_ci * in2_stride_c + h_pos * in2_stride_h + w_pos * in2_stride_w,
                       mask=(global_ci < C2) & ci_mask, other=0.0)
    in3_val = tl.load(in3_ptr + (global_ci - C2) * in3_stride_c + h_pos * in3_stride_h + w_pos * in3_stride_w,
                       mask=(global_ci >= C2) & (global_ci < C2 + C3) & ci_mask, other=0.0)
    conv_val = tl.load(conv_ptr + (global_ci - C2 - C3) * conv_stride_c + h_pos * conv_stride_h + w_pos * conv_stride_w,
                        mask=(global_ci >= C2 + C3) & (global_ci < C_total) & ci_mask, other=0.0)
    
    # Combine: tmp5 value (the transposed cat/reshape result)
    tmp5_val = in2_val + in3_val + conv_val
    
    # Read q_img value: in6[0, head_idx, hw_idx, ci_offsets]
    in6_val = tl.load(in6_ptr + head_idx * in6_stride_h1 + hw_idx * in6_stride_hw + ci_offsets * in6_stride_ci,
                       mask=ci_mask, other=0.0)
    
    # Multiply: tmp6 = in6 * tmp5
    tmp6_val = in6_val * tmp5_val
    
    # Read scale
    scale_val = tl.load(scale_ptr)
    
    # Read factor_att: in4[0, head_idx, hw_idx+1, ci_offsets]
    # Note: we read from row hw_idx+1 because of the padding shift
    # The pad adds a row at position 0, so the factor_att row for output row hw_idx+1
    # corresponds to in4 at row hw_idx+1 (since in4 already has HW+1 rows matching the padded dimension)
    in4_val = tl.load(in4_ptr + head_idx * in4_stride_h1 + (hw_idx + 1) * in4_stride_seq + ci_offsets * in4_stride_ci,
                       mask=ci_mask, other=0.0)
    
    # Write output for row hw_idx+1 (shifted by pad): scale*in4 + tmp6
    out_row = hw_idx + 1
    out_col = head_idx * C_inner + ci_offsets
    
    out_val = scale_val * in4_val + tmp6_val
    
    tl.store(out_ptr + out_row * out_stride_seq + out_col * out_stride_ct,
             out_val, mask=ci_mask & (out_col < C_total))


@triton.jit
def fused_pad_row_zero_kernel(
    out_ptr,
    in4_ptr,
    scale_ptr,
    heads, C_inner,
    in4_stride_h1,
    in4_stride_seq,
    in4_stride_ci,
    out_stride_seq,
    out_stride_ct,
    BLOCK_CI: tl.constexpr,
):
    # Handle row 0 of output: scale * in4[0, head_idx, 0, ci]
    pid = tl.program_id(0)
    head_idx = pid
    
    ci_offsets = tl.arange(0, BLOCK_CI)
    ci_mask = ci_offsets < C_inner
    
    scale_val = tl.load(scale_ptr)
    in4_val = tl.load(in4_ptr + head_idx * in4_stride_h1 + 0 * in4_stride_seq + ci_offsets * in4_stride_ci,
                       mask=ci_mask, other=0.0)
    
    out_col = head_idx * C_inner + ci_offsets
    out_val = scale_val * in4_val
    
    tl.store(out_ptr + 0 * out_stride_seq + out_col * out_stride_ct,
             out_val, mask=ci_mask & (out_col < heads * C_inner))


@torch.fx.wrap
def fused_coat_crpe_dispatch(in_0, in_1, in_2, in_3, in_4, in_5, in_6, route_str):
    """Dispatch wrapper for the full coat CRPE computation."""
    
    # Extract scale and conv2d groups from route string
    # Route format: "route_GROUPS_SCALE" e.g. "route_57_0.22941573387056177"
    parts = route_str.split("_")
    groups = int(parts[1])
    scale = float(parts[2])
    
    # Step 1: Run conv2d (cuDNN is optimized for this)
    conv_out = torch.conv2d(in_5, in_1, in_0, (1, 1), (3, 3), (1, 1), groups)
    
    # Step 2: Run fused post-conv kernel
    return _fused_post_conv_impl(conv_out, in_2, in_3, in_6, in_4, scale)


def _fused_post_conv_impl(conv_out, in_2, in_3, in_6, in_4, scale):
    """
    Fused implementation of:
    cat -> reshape -> transpose -> multiply -> pad -> scale*add -> transpose -> reshape
    """
    C2 = in_2.shape[1]
    C3 = in_3.shape[1]
    C_conv = conv_out.shape[1]
    H = conv_out.shape[2]
    W = conv_out.shape[3]
    heads = in_6.shape[1]
    HW = in_6.shape[2]
    C_inner = in_6.shape[3]
    C_total = C2 + C3 + C_conv
    HW_plus_1 = HW + 1
    
    # Create output tensor: [1, HW+1, C_total]
    out = torch.empty((1, HW_plus_1, C_total), dtype=conv_out.dtype, device=conv_out.device)
    
    # Scale as a tensor (use float32 for precision)
    scale_tensor = torch.tensor([scale], dtype=torch.float32, device=conv_out.device)
    
    # Get strides
    conv_sc, conv_sh, conv_sw = conv_out.stride(1), conv_out.stride(2), conv_out.stride(3)
    in2_sc, in2_sh, in2_sw = in_2.stride(1), in_2.stride(2), in_2.stride(3)
    in3_sc, in3_sh, in3_sw = in_3.stride(1), in_3.stride(2), in_3.stride(3)
    in6_sh1, in6_shw, in6_sci = in_6.stride(1), in_6.stride(2), in_6.stride(3)
    in4_sh1, in4_sseq, in4_sci = in_4.stride(1), in_4.stride(2), in_4.stride(3)
    out_sseq, out_sct = out.stride(1), out.stride(2)
    
    BLOCK_CI = max(1, min(triton.next_power_of_2(C_inner), 2048))
    
    # Launch main kernel for HW rows
    grid_hw = (heads * HW,)
    fused_post_conv_kernel[grid_hw](
        conv_ptr=conv_out, in2_ptr=in_2, in3_ptr=in_3,
        in6_ptr=in_6, in4_ptr=in_4, scale_ptr=scale_tensor,
        out_ptr=out,
        C2=C2, C3=C3, C_conv=C_conv, H=H, W=W,
        heads=heads, C_inner=C_inner, HW=HW, C_total=C_total,
        conv_stride_c=conv_sc, conv_stride_h=conv_sh, conv_stride_w=conv_sw,
        in2_stride_c=in2_sc, in2_stride_h=in2_sh, in2_stride_w=in2_sw,
        in3_stride_c=in3_sc, in3_stride_h=in3_sh, in3_stride_w=in3_sw,
        in6_stride_h1=in6_sh1, in6_stride_hw=in6_shw, in6_stride_ci=in6_sci,
        in4_stride_h1=in4_sh1, in4_stride_seq=in4_sseq, in4_stride_ci=in4_sci,
        out_stride_seq=out_sseq, out_stride_ct=out_sct,
        BLOCK_CI=BLOCK_CI,
    )
    
    # Launch kernel for row 0 (padded row)
    grid_row0 = (heads,)
    fused_pad_row_zero_kernel[grid_row0](
        out_ptr=out, in4_ptr=in_4, scale_ptr=scale_tensor,
        heads=heads, C_inner=C_inner,
        in4_stride_h1=in4_sh1, in4_stride_seq=in4_sseq, in4_stride_ci=in4_sci,
        out_stride_seq=out_sseq, out_stride_ct=out_sct,
        BLOCK_CI=BLOCK_CI,
    )
    
    return (out,)