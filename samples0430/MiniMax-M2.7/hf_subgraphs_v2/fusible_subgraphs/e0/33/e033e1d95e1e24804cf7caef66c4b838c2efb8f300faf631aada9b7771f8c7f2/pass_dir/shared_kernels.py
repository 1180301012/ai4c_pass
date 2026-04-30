import torch
import triton
import triton.language as tl


# ============================================================================
# Kernel 1: Fused Conv2d 1x1 + View
# ============================================================================
@triton.jit
def fused_conv2d_1x1_kernel(
    # Pointers
    input_ptr, weight_ptr, bias_ptr, output_ptr,
    # Strides
    input_batch_stride, input_channel_stride, input_height_stride, input_width_stride,
    weight_out_channel_stride, weight_in_channel_stride,
    output_batch_stride, output_channel_stride, output_spatial_stride,
    # Sizes
    B: tl.constexpr, OUT_C: tl.constexpr, IN_C: tl.constexpr,
    H: tl.constexpr, W: tl.constexpr,
):
    """
    Fused convolution 2D with 1x1 kernel and view to [B, C, H*W].
    Each program computes one (batch, out_channel) pair's output.
    """
    pid = tl.program_id(0)
    
    # Total spatial elements per (batch, out_channel)
    n_spatial = H * W  # 4096
    
    # Calculate which batch and out_channel this program handles
    batch_idx = pid // (OUT_C * n_spatial)
    remainder = pid % (OUT_C * n_spatial)
    out_ch_idx = remainder // n_spatial
    spatial_idx = remainder % n_spatial
    
    # Convert spatial_idx to (h, w)
    h_idx = spatial_idx // W
    w_idx = spatial_idx % W
    
    # Accumulator for convolution: sum over input channels
    acc = tl.load(bias_ptr + out_ch_idx).to(tl.float32)
    
    # Perform the convolution (sum over input channels)
    for ic in range(IN_C):
        # Input offset: [batch, ic, h, w]
        input_offset = (
            batch_idx * input_batch_stride +
            ic * input_channel_stride +
            h_idx * input_height_stride +
            w_idx * input_width_stride
        )
        
        # Weight offset: [out_channel, ic, 0, 0]
        weight_offset = (
            out_ch_idx * weight_out_channel_stride +
            ic * weight_in_channel_stride
        )
        
        inp_val = tl.load(input_ptr + input_offset)
        weight_val = tl.load(weight_ptr + weight_offset)
        
        acc += inp_val.to(tl.float32) * weight_val.to(tl.float32)
    
    # Output offset: [batch, out_channel, spatial_idx]
    output_offset = (
        batch_idx * output_batch_stride +
        out_ch_idx * output_channel_stride +
        spatial_idx * output_spatial_stride
    )
    
    # Store result
    tl.store(output_ptr + output_offset, acc)


@torch.fx.wrap
def fused_conv2d_view_wrapper(x, weight, bias):
    """
    Fused convolution + view operation.
    Input x: [B, IN_CHANNELS, H, W]
    Weight: [OUT_CHANNELS, IN_CHANNELS, 1, 1]
    Bias: [OUT_CHANNELS]
    Returns: [B, OUT_CHANNELS, H*W]
    """
    B, IN_C, H, W = x.shape
    OUT_C = weight.shape[0]
    n_elements = B * OUT_C * H * W
    
    # Output: [B, OUT_C, H*W]
    out = torch.empty((B, OUT_C, H * W), dtype=x.dtype, device=x.device)

    # Grid: one program per output element
    grid = (n_elements,)
    
    fused_conv2d_1x1_kernel[grid](
        x, weight, bias, out,
        x.stride(0), x.stride(1), x.stride(2), x.stride(3),
        weight.stride(0), weight.stride(1),
        out.stride(0), out.stride(1), out.stride(2),
        B, OUT_C, IN_C, H, W,
    )
    
    return out


# ============================================================================
# Kernel 2: Mean Reduction over dim=-2 (optimized with blocked reduction)
# ============================================================================
@triton.jit
def mean_dim_minus2_kernel(
    input_ptr,
    output_ptr,
    stride_b, stride_h, stride_c,
    B: tl.constexpr, H: tl.constexpr, C: tl.constexpr,
    BLOCK_SIZE_H: tl.constexpr,
):
    """
    Compute mean over dim=-2 (the middle dimension) with blocked reduction.
    Input shape: [B, H, C]
    Output shape: [B, 1, C]
    
    Each program handles one (batch, channel) element.
    Threads cooperate to reduce over H using a blocked approach.
    """
    pid = tl.program_id(0)
    
    # Calculate batch and channel indices
    batch_idx = pid // C
    channel_idx = pid % C
    
    # Parallel reduction over H with strided access
    # Each thread handles a subset of H elements with stride = BLOCK_SIZE_H
    acc = 0.0
    h = tl.program_id(0) % BLOCK_SIZE_H
    
    for i in range(BLOCK_SIZE_H):
        actual_h = h + i * BLOCK_SIZE_H
        if actual_h < H:
            offset = batch_idx * stride_b + actual_h * stride_h + channel_idx
            val = tl.load(input_ptr + offset)
            acc += val.to(tl.float32)
    
    # Compute mean
    acc = acc / H
    
    # Store result at output offset
    out_offset = batch_idx * C + channel_idx
    tl.store(output_ptr + out_offset, acc)


@torch.fx.wrap
def mean_dim_minus2_wrapper(x):
    """
    Compute mean over dim=-2 (keepdim=True).
    Input: [B, H, C]
    Output: [B, 1, C]
    """
    B, H, C = x.shape
    
    # Output: [B, 1, C]
    out = torch.empty((B, 1, C), dtype=x.dtype, device=x.device)
    
    n_elements = B * C
    
    # Use blocked size that divides H well
    BLOCK_SIZE_H = min(64, H)
    if BLOCK_SIZE_H == 0:
        BLOCK_SIZE_H = 64
    
    mean_dim_minus2_kernel[(n_elements,)](
        x, out,
        x.stride(0), x.stride(1), x.stride(2),
        B, H, C,
        BLOCK_SIZE_H,
    )
    
    return out


# ============================================================================
# Shared Dispatch Function
# ============================================================================
@torch.fx.wrap
def shared_dispatch(*args):
    """
    Shared dispatch function that routes to the appropriate kernel based on route.
    Last argument is the route string.
    """
    route = args[-1]
    inner_args = args[:-1]
    
    if route == "conv2d_view":
        return fused_conv2d_view_wrapper(*inner_args)
    elif route == "mean_dim_minus2":
        return mean_dim_minus2_wrapper(*inner_args)
    else:
        raise ValueError(f"Unknown route: {route}")