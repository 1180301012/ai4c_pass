import torch
import triton
import triton.language as tl


def pattern(x, reshape_dim0, reshape_dim2, reshape_dim3):
    """
    Match the pattern: reshape + interpolate
    Pattern matches reshape dimensions that produce 16x16 spatial size and then 
    bilinear interpolate to 128x128.
    
    Args:
        x: Input tensor after linear and permute, shape (B, 768, 256) where 256=16*16
        reshape_dim0: First dimension of reshape (e.g., 2, 8, 12, 24, 64)
        reshape_dim2: Should be 16
        reshape_dim3: Should be 16
    """
    tmp_4 = x.reshape(reshape_dim0, -1, reshape_dim2, reshape_dim3)
    tmp_5 = torch.nn.functional.interpolate(
        tmp_4, size=(128, 128), mode='bilinear', align_corners=False
    )
    return tmp_5


def replacement_args(x, reshape_dim0, reshape_dim2, reshape_dim3):
    """Extract arguments for the replacement function."""
    return (x, reshape_dim0, reshape_dim2, reshape_dim3)


@triton.jit
def fused_bilinear_interpolate_kernel(
    x_ptr,
    output_ptr,
    B,  # batch dimension
    C,  # channels = 768
    stride_x_batch,
    stride_x_channel,
    stride_x_h,
    stride_x_w,
    stride_out_batch,
    stride_out_channel,
    stride_out_h,
    stride_out_w,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fused kernel that performs bilinear interpolation from 16x16 to 128x128.
    
    Input x has shape (B, C, 16, 16) after reshape.
    Output has shape (B, C, 128, 128).
    
    Grid: (B * C, 128) where 128 is output height
    """
    # pid_batch_channel identifies which (batch, channel) pair we're processing
    pid_batch_channel = tl.program_id(0)
    # pid_h is the output height index
    pid_h = tl.program_id(1)
    
    batch_idx = pid_batch_channel // C
    channel_idx = pid_batch_channel % C
    
    # Output height range for this program
    h_start = pid_h * BLOCK_SIZE
    h_end = min(h_start + BLOCK_SIZE, 128)
    
    # Create height offsets
    offs_h = h_start + tl.arange(0, BLOCK_SIZE)
    mask_h = offs_h < 128
    
    # Create width offsets (full 128 width)
    offs_w = tl.arange(0, 128)
    mask_w = offs_w < 128
    
    # Map output coordinates to input coordinates (scaling by 16/128 = 1/8)
    h_input = offs_h.to(tl.float32) / 128.0 * 16.0
    w_input = offs_w.to(tl.float32) / 128.0 * 16.0
    
    # Get integer coordinates for the 4 neighbors
    h0 = tl.floor(h_input).to(tl.int32)
    w0 = tl.floor(w_input).to(tl.int32)
    h1 = h0 + 1
    w1 = w0 + 1
    
    # Clamp to valid range [0, 15]
    h0 = tl.clip(h0, 0, 15)
    w0 = tl.clip(w0, 0, 15)
    h1 = tl.clip(h1, 0, 15)
    w1 = tl.clip(w1, 0, 15)
    
    # Compute fractional offsets for interpolation
    h_frac = h_input - tl.floor(h_input)
    w_frac = w_input - tl.floor(w_input)
    
    # Reshape for broadcasting: (BLOCK_SIZE, 1) and (1, 128)
    h_frac = h_frac[:, None]
    w_frac = w_frac[None, :]
    
    # Base offsets for x (the 16x16 input)
    x_base = (batch_idx * stride_x_batch + 
              channel_idx * stride_x_channel)
    
    # Load the 4 corner values for bilinear interpolation
    # Corner 00: (h0, w0)
    x_offset_00 = x_base + h0[:, None] * stride_x_h + w0[None, :] * stride_x_w
    v00 = tl.load(x_ptr + x_offset_00, mask=mask_h[:, None] & mask_w[None, :], other=0.0)
    
    # Corner 01: (h0, w1)
    x_offset_01 = x_base + h0[:, None] * stride_x_h + w1[None, :] * stride_x_w
    v01 = tl.load(x_ptr + x_offset_01, mask=mask_h[:, None] & mask_w[None, :], other=0.0)
    
    # Corner 10: (h1, w0)
    x_offset_10 = x_base + h1[:, None] * stride_x_h + w0[None, :] * stride_x_w
    v10 = tl.load(x_ptr + x_offset_10, mask=mask_h[:, None] & mask_w[None, :], other=0.0)
    
    # Corner 11: (h1, w1)
    x_offset_11 = x_base + h1[:, None] * stride_x_h + w1[None, :] * stride_x_w
    v11 = tl.load(x_ptr + x_offset_11, mask=mask_h[:, None] & mask_w[None, :], other=0.0)
    
    # Bilinear interpolation formula:
    # out = v00 * (1-h) * (1-w) + v01 * (1-h) * w + v10 * h * (1-w) + v11 * h * w
    out = (v00 * (1.0 - h_frac) * (1.0 - w_frac) +
           v01 * (1.0 - h_frac) * w_frac +
           v10 * h_frac * (1.0 - w_frac) +
           v11 * h_frac * w_frac)
    
    # Compute output offsets
    out_offset = (batch_idx * stride_out_batch + 
                  channel_idx * stride_out_channel + 
                  offs_h[:, None] * stride_out_h + 
                  offs_w[None, :] * stride_out_w)
    
    # Store the result
    tl.store(output_ptr + out_offset, out, mask=mask_h[:, None] & mask_w[None, :])


@torch.fx.wrap
def fused_reshape_interpolate_kernel_wrapper(x, reshape_dim0, reshape_dim2, reshape_dim3):
    """
    Wrapper for the fused reshape+interpolate Triton kernel.
    
    Takes input tensor x with shape (B, 768, 256) (after linear and permute)
    and performs:
    1. reshape to (reshape_dim0, 768, 16, 16) where reshape_dim0 is the batch dimension
    2. bilinear interpolation from (16, 16) to (128, 128)
    
    Output shape: (reshape_dim0, 768, 128, 128)
    """
    B, C, N = x.shape  # x shape is (B, 768, 256)
    
    assert N == 256, f"Expected N=256 (16*16), got N={N}"
    assert reshape_dim2 == 16 and reshape_dim3 == 16, f"Only 16x16 input supported, got ({reshape_dim2}, {reshape_dim3})"
    assert C == 768, f"Expected C=768, got C={C}"
    
    batch = reshape_dim0  # Output batch dimension
    out_channels = C  # 768 channels
    
    # Reshape input to (batch, 768, 16, 16)
    x_reshaped = x.view(batch, out_channels, 16, 16)
    
    # Create output tensor with shape (batch, 768, 128, 128)
    output = torch.empty((batch, out_channels, 128, 128), dtype=x.dtype, device=x.device)
    
    # Strides for reshaped input
    stride_x_batch = x_reshaped.stride(0)
    stride_x_channel = x_reshaped.stride(1)
    stride_x_h = x_reshaped.stride(2)
    stride_x_w = x_reshaped.stride(3)
    
    # Strides for output
    stride_out_batch = output.stride(0)
    stride_out_channel = output.stride(1)
    stride_out_h = output.stride(2)
    stride_out_w = output.stride(3)
    
    # Grid: (batch * channels, 128/block_size)
    # 128/block_size is the number of blocks in the height dimension
    BLOCK_SIZE = 16
    grid_h = (128 + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Launch kernel with grid (batch * channels, grid_h)
    fused_bilinear_interpolate_kernel[(batch * out_channels, grid_h)](
        x_reshaped,
        output,
        batch,
        out_channels,
        stride_x_batch,
        stride_x_channel,
        stride_x_h,
        stride_x_w,
        stride_out_batch,
        stride_out_channel,
        stride_out_h,
        stride_out_w,
        BLOCK_SIZE,
    )
    
    return output


def replacement_func():
    """Return the replacement function."""
    return fused_reshape_interpolate_kernel_wrapper