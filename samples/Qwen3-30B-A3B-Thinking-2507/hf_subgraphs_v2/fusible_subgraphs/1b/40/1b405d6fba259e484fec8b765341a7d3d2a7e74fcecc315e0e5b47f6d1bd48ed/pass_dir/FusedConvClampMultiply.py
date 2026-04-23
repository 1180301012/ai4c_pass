import torch
import triton
import triton.language as tl

def pattern(in_2, in_1, in_0, in_3):
    conv = torch.conv2d(in_2, in_1, in_0, (1, 1), (0, 0), (1, 1), 1)
    clamp = torch.nn.functional.hardtanh(in_3, 0.0, 6.0, False)
    result = clamp * conv
    return result

def replacement_args(in_2, in_1, in_0, in_3):
    return (in_2, in_1, in_0, in_3)

@triton.jit
def fused_conv_multiply_kernel(
    in_2_ptr, in_1_ptr, in_0_ptr, in_3_ptr, out_ptr,
    in_channels, out_channels, H, W, batch,
    BLOCK_OUT_CH: tl.constexpr, BLOCK_H: tl.constexpr, BLOCK_W: tl.constexpr
):
    # Calculate grid indices
    out_ch_start = tl.program_id(0) * BLOCK_OUT_CH
    h_start = tl.program_id(1) * BLOCK_H
    w_start = tl.program_id(2) * BLOCK_W

    # Create masks for boundaries
    out_ch_mask = out_ch_start + tl.arange(0, BLOCK_OUT_CH) < out_channels
    h_mask = h_start + tl.arange(0, BLOCK_H) < H
    w_mask = w_start + tl.arange(0, BLOCK_W) < W

    # Load weights for current output channels block
    weights_base = in_1_ptr + out_ch_start * in_channels
    weights = tl.load(
        weights_base + (tl.arange(0, BLOCK_OUT_CH)[:, None] * in_channels) + 
        tl.arange(0, in_channels)[None, :],
        mask=(tl.arange(0, BLOCK_OUT_CH)[:, None] < out_channels) & 
              (tl.arange(0, in_channels)[None, :] < in_channels),
        other=0.0
    )

    # Initialize accumulator
    acc = tl.zeros((BLOCK_OUT_CH, BLOCK_H, BLOCK_W), dtype=tl.float32)

    # Process all input channels
    for c in range(in_channels):
        # Load input tensor for current channel
        in2_val = tl.load(
            in_2_ptr + (tl.arange(0, batch)[:, None, None] * in_channels * H * W) + 
                     (c * H * W) + 
                     (h_start + tl.arange(0, BLOCK_H))[None, :, None] * W + 
                     (w_start + tl.arange(0, BLOCK_W))[None, None, :],
            mask=(tl.arange(0, batch)[:, None, None] < batch) & 
                  h_mask[None, :, None] & 
                  w_mask[None, None, :],
            other=0.0
        )
        
        # Multiply with weights for current channel
        acc += in2_val * weights[:, :, None, None]

    # Add bias
    bias = tl.load(
        in_0_ptr + out_ch_start + tl.arange(0, BLOCK_OUT_CH),
        mask=out_ch_mask,
        other=0.0
    )
    acc = acc + bias[:, None, None]

    # Clamp in_3 values and multiply
    in3_val = tl.load(
        in_3_ptr + (tl.arange(0, batch)[:, None, None] * out_channels * H * W) + 
                 (out_ch_start + tl.arange(0, BLOCK_OUT_CH))[None, :, None] * H * W + 
                 (h_start + tl.arange(0, BLOCK_H))[None, None, :] * W + 
                 (w_start + tl.arange(0, BLOCK_W))[None, None, :],
        mask=(tl.arange(0, batch)[:, None, None] < batch) & 
              out_ch_mask[None, :, None] & 
              h_mask[None, None, :] & 
              w_mask[None, None, :],
        other=0.0
    )
    in3_clamped = tl.minimum(6.0, tl.maximum(0.0, in3_val))
    out = acc * in3_clamped

    # Store result
    tl.store(
        out_ptr + (tl.arange(0, batch)[:, None, None] * out_channels * H * W) + 
                 (out_ch_start + tl.arange(0, BLOCK_OUT_CH))[None, :, None] * H * W + 
                 (h_start + tl.arange(0, BLOCK_H))[None, None, :] * W + 
                 (w_start + tl.arange(0, BLOCK_W))[None, None, :],
        out,
        mask=(tl.arange(0, batch)[:, None, None] < batch) & 
              out_ch_mask[None, :, None] & 
              h_mask[None, None, :] & 
              w_mask[None, None, :]
    )

@torch.fx.wrap
def kernel_wrapper(in_2, in_1, in_0, in_3):
    batch, in_channels, H, W = in_2.shape
    out_channels = in_1.shape[0]
    out = torch.empty((batch, out_channels, H, W), dtype=in_2.dtype, device=in_2.device)
    
    BLOCK_OUT_CH = 32
    BLOCK_H = 16
    BLOCK_W = 16
    
    grid_h = (out_channels + BLOCK_OUT_CH - 1) // BLOCK_OUT_CH
    grid_w = (H + BLOCK_H - 1) // BLOCK_H
    grid = (grid_h, grid_w)
    
    fused_conv_multiply_kernel[grid](
        in_2, in_1, in_0, in_3, out,
        in_channels, out_channels, H, W, batch,
        BLOCK_OUT_CH, BLOCK_H, BLOCK_W
    )
    
    return out

def replacement_func():
    return kernel_wrapper