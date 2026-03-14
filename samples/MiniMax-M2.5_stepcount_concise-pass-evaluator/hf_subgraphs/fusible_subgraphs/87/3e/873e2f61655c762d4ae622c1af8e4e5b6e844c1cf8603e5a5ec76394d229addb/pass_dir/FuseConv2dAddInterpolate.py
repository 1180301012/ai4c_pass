import torch
import triton
import triton.language as tl


# Pattern matching function - matches Conv2D + Add + Interpolate
def pattern(in_0, in_5, in_6):
    """
    Match the pattern:
    tmp_5 = torch.conv2d(in_6, in_0, None, (1, 1), (0, 0), (1, 1), 1)
    tmp_6 = in_5 + tmp_5
    tmp_7 = torch.nn.functional.interpolate(tmp_6, (64, 64), None, 'bilinear', False)
    
    Returns both tmp_6 (for next op) and tmp_7 (interpolated result)
    """
    tmp_5 = torch.conv2d(in_6, in_0, None, (1, 1), (0, 0), (1, 1), 1)
    tmp_6 = in_5 + tmp_5
    tmp_7 = torch.nn.functional.interpolate(tmp_6, (64, 64), None, 'bilinear', False)
    return tmp_6, tmp_7


def replacement_args(in_0, in_5, in_6):
    return (in_0, in_5, in_6)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 256}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 128}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 32}, num_stages=3, num_warps=4),
    ],
    key=['B', 'C_out', 'H_in', 'W_in', 'H_out', 'W_out'],
)
@triton.jit
def conv_add_interpolate_kernel(
    # Conv weights
    weight_ptr,
    # Input feature maps
    in_5_ptr,  # [B, C_out, H_in, W_in]
    in_6_ptr,  # [B, C_in, H_in, W_in]
    # Output
    output_ptr,
    # Sizes
    B, C_out, C_in, H_in, W_in, H_out, W_out,
    # Strides
    stride_weight, stride_in_5, stride_in_6, stride_out,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """
    Fused kernel that performs:
    1. Conv2D with 1x1 kernel (essentially channel-wise projection)
    2. Add in_5
    3. Bilinear interpolate from H_in x W_in to H_out x W_out
    """
    # Get program IDs
    pid_b = tl.program_id(0)
    pid_c = tl.program_id(1)
    
    # Calculate output position
    off_b = pid_b
    off_c = pid_c
    
    # Input channel offset for weight
    weight_offset = off_c * C_in
    
    # Each program processes one output channel for one batch
    # The loop is over output spatial locations (interpolated)
    
    # Compute interpolation weights
    # Bilinear interpolation: scale factors
    scale_h = (H_in - 1) / (H_out - 1) if H_out > 1 else 0.0
    scale_w = (W_in - 1) / (W_out - 1) if W_out > 1 else 0.0
    
    # Load weight for this output channel [C_in]
    # Weight shape: [C_out, C_in, 1, 1], stride = C_in
    weight_offsets = weight_offset + tl.arange(0, C_in)
    weight = tl.load(weight_ptr + weight_offsets)
    
    # Process output spatial locations in blocks
    for h_out_start in range(0, H_out, BLOCK_M):
        for w_out_start in range(0, W_out, BLOCK_N):
            # Calculate output offsets
            h_out_offs = h_out_start + tl.arange(0, BLOCK_M)
            w_out_offs = w_out_start + tl.arange(0, BLOCK_N)
            
            # Mask for valid outputs
            mask_h = h_out_offs < H_out
            mask_w = w_out_offs < W_out
            mask = mask_h & mask_w
            
            # Calculate source coordinates for bilinear interpolation
            # h_src = h_out * scale_h, w_src = w_out * scale_w
            h_src = (h_out_offs * scale_h).to(tl.float32)
            w_src = (w_out_offs * scale_w).to(tl.float32)
            
            # Floor and ceil for bilinear interpolation
            h0 = tl.floor(h_src).to(tl.int32)
            h1 = h0 + 1
            w0 = tl.floor(w_src).to(tl.int32)
            w1 = w0 + 1
            
            # Clamp to valid range
            h0 = tl.maximum(tl.minimum(h0, H_in - 1), 0)
            h1 = tl.maximum(tl.minimum(h1, H_in - 1), 0)
            w0 = tl.maximum(tl.minimum(w0, W_in - 1), 0)
            w1 = tl.maximum(tl.minimum(w1, W_in - 1), 0)
            
            # Interpolation weights
            h1_weight = h_src - h0.to(tl.float32)
            w1_weight = w_src - w0.to(tl.float32)
            h0_weight = 1.0 - h1_weight
            w0_weight = 1.0 - w1_weight
            
            # Compute convolution + add + interpolate for each corner
            # Conv is essentially: out[b, c, h, w] = sum_c(in[b, c_in, h, w] * weight[c, c_in])
            # For 1x1 conv, spatial position is same
            
            # Load in_5 [B, C_out, H_in, W_in]
            # For bilinear interpolation, we need 4 samples
            # Each sample needs the conv result at that position plus in_5
            
            # We need to compute conv for each input position and then interpolate
            # Since conv is 1x1, conv[b, c, h, w] = sum_c(in[b, c_in, h, w] * weight[c, c_in])
            
            # Let's accumulate using a loop over C_in
            acc_00 = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
            acc_01 = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
            acc_10 = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
            acc_11 = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
            
            for c_in_start in range(0, C_in, 64):
                c_in_offs = c_in_start + tl.arange(0, 64)
                mask_c = c_in_offs < C_in
                
                # Load weight slice
                w_offsets = weight_offset + c_in_offs
                w_slice = tl.load(weight_ptr + w_offsets, mask=mask_c, other=0.0)
                
                # Load in_6 for 4 corners
                # (h0, w0)
                off_in6_00 = (off_b * C_in * H_in * W_in + 
                              c_in_offs[:, None, None] * stride_in_6 + 
                              h0[None, :, None] * W_in + 
                              w0[None, None, :])
                in6_00 = tl.load(in_6_ptr + off_in6_00, mask=mask_c[:, None, None] & mask_h[None, :, None] & mask_w[None, None, :], other=0.0)
                
                # (h0, w1)
                off_in6_01 = (off_b * C_in * H_in * W_in + 
                              c_in_offs[:, None, None] * stride_in_6 + 
                              h0[None, :, None] * W_in + 
                              w1[None, None, :])
                in6_01 = tl.load(in_6_ptr + off_in6_01, mask=mask_c[:, None, None] & mask_h[None, :, None] & mask_w[None, None, :], other=0.0)
                
                # (h1, w0)
                off_in6_10 = (off_b * C_in * H_in * W_in + 
                              c_in_offs[:, None, None] * stride_in_6 + 
                              h1[None, :, None] * W_in + 
                              w0[None, None, :])
                in6_10 = tl.load(in_6_ptr + off_in6_10, mask=mask_c[:, None, None] & mask_h[None, :, None] & mask_w[None, None, :], other=0.0)
                
                # (h1, w1)
                off_in6_11 = (off_b * C_in * H_in * W_in + 
                              c_in_offs[:, None, None] * stride_in_6 + 
                              h1[None, :, None] * W_in + 
                              w1[None, None, :])
                in6_11 = tl.load(in_6_ptr + off_in6_11, mask=mask_c[:, None, None] & mask_h[None, :, None] & mask_w[None, None, :], other=0.0)
                
                # Multiply by weight and accumulate
                acc_00 += tl.sum(w_slice[:, None, None] * in6_00, axis=0)
                acc_01 += tl.sum(w_slice[:, None, None] * in6_01, axis=0)
                acc_10 += tl.sum(w_slice[:, None, None] * in6_10, axis=0)
                acc_11 += tl.sum(w_slice[:, None, None] * in6_11, axis=0)
            
            # Now add in_5 at the source positions and interpolate
            # For each of the 4 corners, we need to add in_5 at that position
            
            # Load in_5 at corner positions
            # in_5 shape: [B, C_out, H_in, W_in]
            off_in5_00 = (off_b * C_out * H_in * W_in + 
                          off_c * H_in * W_in + 
                          h0 * W_in + w0)
            in5_00 = tl.load(in_5_ptr + off_in5_00, mask=mask_h & mask_w, other=0.0)
            
            off_in5_01 = (off_b * C_out * H_in * W_in + 
                          off_c * H_in * W_in + 
                          h0 * W_in + w1)
            in5_01 = tl.load(in_5_ptr + off_in5_01, mask=mask_h & mask_w, other=0.0)
            
            off_in5_10 = (off_b * C_out * H_in * W_in + 
                          off_c * H_in * W_in + 
                          h1 * W_in + w0)
            in5_10 = tl.load(in_5_ptr + off_in5_10, mask=mask_h & mask_w, other=0.0)
            
            off_in5_11 = (off_b * C_out * H_in * W_in + 
                          off_c * H_in * W_in + 
                          h1 * W_in + w1)
            in5_11 = tl.load(in_5_ptr + off_in5_11, mask=mask_h & mask_w, other=0.0)
            
            # Add in_5 to conv results
            conv_plus_in5_00 = acc_00 + in5_00
            conv_plus_in5_01 = acc_01 + in5_01
            conv_plus_in5_10 = acc_10 + in5_10
            conv_plus_in5_11 = acc_11 + in5_11
            
            # Bilinear interpolate
            # out = h0_weight * w0_weight * conv_plus_in5_00 + 
            #       h0_weight * w1_weight * conv_plus_in5_01 +
            #       h1_weight * w0_weight * conv_plus_in5_10 +
            #       h1_weight * w1_weight * conv_plus_in5_11
            out = (h0_weight[:, None] * w0_weight[None, :] * conv_plus_in5_00 +
                   h0_weight[:, None] * w1_weight[None, :] * conv_plus_in5_01 +
                   h1_weight[:, None] * w0_weight[None, :] * conv_plus_in5_10 +
                   h1_weight[:, None] * w1_weight[None, :] * conv_plus_in5_11)
            
            # Store output [B, C_out, H_out, W_out]
            out_offsets = (off_b * C_out * H_out * W_out + 
                           off_c * H_out * W_out + 
                           h_out_offs[:, None] * W_out + 
                           w_out_offs[None, :])
            tl.store(output_ptr + out_offsets, out, mask=mask_h[:, None] & mask_w[None, :])


def conv_add_interpolate(weight, in_5, in_6):
    """
    Fused Conv2D + Add + Interpolate operation.
    
    Args:
        weight: [C_out, C_in, 1, 1] - Convolution weights
        in_5: [B, C_out, H_in, W_in] - Input to add
        in_6: [B, C_in, H_in, W_in] - Input to convolve
    
    Returns:
        [B, C_out, H_out, W_out] - Interpolated result
    """
    B, C_in, H_in, W_in = in_6.shape
    C_out, _, _, _ = weight.shape
    H_out, W_out = 64, 64  # Target size from interpolate
    
    # Output
    output = torch.empty((B, C_out, H_out, W_out), device=in_6.device, dtype=torch.float32)
    
    # Strides
    stride_weight = weight.stride(0)
    stride_in_5 = in_5.stride(0)
    stride_in_6 = in_6.stride(0)
    stride_out = output.stride(0)
    
    # Grid: (B, C_out)
    grid = (B, C_out)
    
    conv_add_interpolate_kernel[grid](
        weight,
        in_5,
        in_6,
        output,
        B, C_out, C_in, H_in, W_in, H_out, W_out,
        stride_weight, stride_in_5, stride_in_6, stride_out,
    )
    
    return output


@torch.fx.wrap
def conv_add_interpolate_wrapper(weight, in_5, in_6):
    return conv_add_interpolate(weight, in_5, in_6)


def replacement_func():
    return conv_add_interpolate_wrapper