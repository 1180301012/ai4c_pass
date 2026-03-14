import torch
import triton
import triton.language as tl


def pattern(in_0, in_1):
    """
    Pattern matching cat + slice + mean operation.
    The slice operation takes all channels after concatenation (no-op).
    """
    tmp_0 = torch.cat([in_0, in_1], dim=1)
    # The slice takes channels [0:C_total] where C_total = C0 + C1
    # This matches any slice that takes all channels
    tmp_1 = tmp_0[:, :, :, :]
    tmp_2 = tmp_1.mean((2, 3), keepdim=True)
    return (tmp_1, tmp_2)


def replacement_args(in_0, in_1):
    return (in_0, in_1)


@triton.jit
def cat_mean_kernel(
    in_0_ptr, in_1_ptr, 
    out_cat_ptr, out_mean_ptr,
    batch, c0, c1, height, width,
    stride_b0, stride_c0, stride_h0, stride_w0,
    stride_b1, stride_c1, stride_h1, stride_w1,
    stride_bc, stride_cc, stride_hc, stride_wc,
    stride_bm, stride_cm, stride_hm, stride_wm,
    BLOCK_C0: tl.constexpr, BLOCK_C1: tl.constexpr,
    BLOCK_H: tl.constexpr, BLOCK_W: tl.constexpr,
):
    """
    Fused kernel for cat + mean.
    Each program handles one (batch, channel) element.
    """
    pid_b = tl.program_id(0)
    pid_c = tl.program_id(1)
    
    c_total = c0 + c1
    spatial_size = height * width
    
    # Determine which input this channel comes from
    is_from_in_0 = pid_c < c0
    
    # Compute mean for this (batch, channel) across spatial dimensions
    acc = 0.0
    
    # Load and accumulate over spatial dimensions
    for h_start in range(0, height, BLOCK_H):
        h_end = tl.minimum(h_start + BLOCK_H, height)
        for w_start in range(0, width, BLOCK_W):
            w_end = tl.minimum(w_start + BLOCK_W, width)
            
            h_offsets = h_start + tl.arange(0, BLOCK_H)
            w_offsets = w_start + tl.arange(0, BLOCK_W)
            
            h_mask = h_offsets < h_end
            w_mask = w_offsets < w_end
            
            # Load values based on source
            if is_from_in_0:
                # Channel from in_0
                c_idx = pid_c
                for h_idx in range(BLOCK_H):
                    if h_start + h_idx < height:
                        h = h_start + h_idx
                        for w_idx in range(BLOCK_W):
                            if w_start + w_idx < width:
                                w = w_start + w_idx
                                offset = pid_b * stride_b0 + c_idx * stride_c0 + h * stride_h0 + w * stride_w0
                                val = tl.load(in_0_ptr + offset)
                                acc += val
                                
                                # Store to output concatenated tensor
                                out_offset = pid_b * stride_bc + pid_c * stride_cc + h * stride_hc + w * stride_wc
                                tl.store(out_cat_ptr + out_offset, val)
            else:
                # Channel from in_1
                c_idx = pid_c - c0
                for h_idx in range(BLOCK_H):
                    if h_start + h_idx < height:
                        h = h_start + h_idx
                        for w_idx in range(BLOCK_W):
                            if w_start + w_idx < width:
                                w = w_start + w_idx
                                offset = pid_b * stride_b1 + c_idx * stride_c1 + h * stride_h1 + w * stride_w1
                                val = tl.load(in_1_ptr + offset)
                                acc += val
                                
                                # Store to output concatenated tensor
                                out_offset = pid_b * stride_bc + pid_c * stride_cc + h * stride_hc + w * stride_wc
                                tl.store(out_cat_ptr + out_offset, val)
    
    # Compute mean and store
    mean_val = acc / spatial_size
    mean_offset = pid_b * stride_bm + pid_c * stride_cm
    tl.store(out_mean_ptr + mean_offset, mean_val)


@torch.fx.wrap
def fused_cat_mean(in_0, in_1):
    """
    Fused implementation of cat + mean operation.
    """
    batch, c0, height, width = in_0.shape
    c1 = in_1.shape[1]
    c_total = c0 + c1
    
    # Output tensors
    out_cat = torch.empty((batch, c_total, height, width), dtype=in_0.dtype, device=in_0.device)
    out_mean = torch.empty((batch, c_total, 1, 1), dtype=in_0.dtype, device=in_0.device)
    
    # Launch kernel
    BLOCK_C0 = 1
    BLOCK_C1 = 1
    BLOCK_H = 32
    BLOCK_W = 32
    
    grid = (batch, c_total)
    
    cat_mean_kernel[grid](
        in_0, in_1,
        out_cat, out_mean,
        batch, c0, c1, height, width,
        in_0.stride(0), in_0.stride(1), in_0.stride(2), in_0.stride(3),
        in_1.stride(0), in_1.stride(1), in_1.stride(2), in_1.stride(3),
        out_cat.stride(0), out_cat.stride(1), out_cat.stride(2), out_cat.stride(3),
        out_mean.stride(0), out_mean.stride(1), out_mean.stride(2), out_mean.stride(3),
        BLOCK_C0, BLOCK_C1, BLOCK_H, BLOCK_W,
    )
    
    return out_cat, out_mean


def replacement_func():
    return fused_cat_mean