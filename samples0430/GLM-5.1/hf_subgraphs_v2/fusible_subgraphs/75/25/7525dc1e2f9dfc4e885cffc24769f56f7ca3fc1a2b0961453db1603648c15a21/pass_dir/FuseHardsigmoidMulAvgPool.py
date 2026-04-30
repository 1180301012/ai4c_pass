import torch
import triton
import triton.language as tl


def pattern(conv2d_output, in_2):
    tmp_3 = torch.nn.functional.hardsigmoid(conv2d_output, False)
    tmp_4 = in_2 * tmp_3
    tmp_5 = torch.nn.functional.adaptive_avg_pool2d(tmp_4, 1)
    tmp_6 = tmp_5.flatten(1, -1)
    tmp_7 = torch.nn.functional.dropout(tmp_6, 0.0, False, False)
    return tmp_7


def replacement_args(conv2d_output, in_2):
    return (conv2d_output, in_2)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_C': 8, 'BLOCK_HW': 32}, num_warps=2),
        triton.Config({'BLOCK_C': 8, 'BLOCK_HW': 64}, num_warps=4),
        triton.Config({'BLOCK_C': 16, 'BLOCK_HW': 32}, num_warps=4),
        triton.Config({'BLOCK_C': 16, 'BLOCK_HW': 64}, num_warps=4),
        triton.Config({'BLOCK_C': 16, 'BLOCK_HW': 128}, num_warps=4),
        triton.Config({'BLOCK_C': 32, 'BLOCK_HW': 32}, num_warps=4),
        triton.Config({'BLOCK_C': 32, 'BLOCK_HW': 64}, num_warps=4),
        triton.Config({'BLOCK_C': 32, 'BLOCK_HW': 128}, num_warps=8),
        triton.Config({'BLOCK_C': 64, 'BLOCK_HW': 32}, num_warps=8),
        triton.Config({'BLOCK_C': 64, 'BLOCK_HW': 64}, num_warps=8),
        triton.Config({'BLOCK_C': 64, 'BLOCK_HW': 128}, num_warps=8),
    ],
    key=['channels', 'height', 'width'],
)
@triton.jit
def fused_hardsigmoid_mul_avgpool_kernel(
    conv_out_ptr, x_ptr, out_ptr,
    batch_size: tl.int32, channels: tl.int32, height: tl.int32, width: tl.int32,
    conv_stride_b: tl.int32, conv_stride_c: tl.int32,
    x_stride_b: tl.int32, x_stride_c: tl.int32, x_stride_h: tl.int32, x_stride_w: tl.int32,
    out_stride_b: tl.int32, out_stride_c: tl.int32,
    BLOCK_C: tl.constexpr,
    BLOCK_HW: tl.constexpr,
):
    pid = tl.program_id(0)
    num_c_blocks = tl.cdiv(channels, BLOCK_C)
    b = pid // num_c_blocks
    c_block_idx = pid % num_c_blocks
    c_start = c_block_idx * BLOCK_C
    c_offsets = c_start + tl.arange(0, BLOCK_C)
    c_mask = c_offsets < channels
    
    # Load conv2d output values for BLOCK_C channels at once
    # conv_out has shape [B, C, 1, 1], accessing [b, c_offsets, 0, 0]
    conv_vals = tl.load(conv_out_ptr + b * conv_stride_b + c_offsets * conv_stride_c, mask=c_mask, other=0.0)
    conv_vals_f32 = tl.cast(conv_vals, tl.float32)
    
    # hardsigmoid: clamp(x + 3, 0, 6) / 6
    hs = conv_vals_f32 + 3.0
    hs = tl.where(hs < 0.0, 0.0, hs)
    hs = tl.where(hs > 6.0, 6.0, hs)
    hs = hs / 6.0
    
    # Compute mean of x[b, c_offsets, :, :] for each channel
    # Load 2D block [BLOCK_C, BLOCK_HW] and sum over spatial dimension
    spatial_sums = tl.zeros([BLOCK_C], tl.float32)
    hw_size = height * width
    hw_size_float = tl.cast(hw_size, tl.float32)
    
    for hw_start in tl.range(0, hw_size, BLOCK_HW):
        hw_offsets = hw_start + tl.arange(0, BLOCK_HW)
        hw_mask = hw_offsets < hw_size
        
        h = hw_offsets // width
        w = hw_offsets % width
        
        # Load x[b, c_offsets, h, w] -> [BLOCK_C, BLOCK_HW]
        x_ptrs = x_ptr + b * x_stride_b + c_offsets[:, None] * x_stride_c + h[None, :] * x_stride_h + w[None, :] * x_stride_w
        x_vals = tl.load(x_ptrs, mask=c_mask[:, None] & hw_mask[None, :], other=0.0)
        x_vals_f32 = tl.cast(x_vals, tl.float32)
        spatial_sums += tl.sum(x_vals_f32, axis=1)  # Sum over spatial dimension
    
    mean_vals = spatial_sums / hw_size_float
    
    # Final result: hardsigmoid(conv_val) * mean(x[b,c,:,:]) for each channel
    out_vals = hs * mean_vals
    
    # Store results
    out_ptrs = out_ptr + b * out_stride_b + c_offsets * out_stride_c
    tl.store(out_ptrs, out_vals, mask=c_mask)


@torch.fx.wrap
def fused_hardsigmoid_mul_avgpool(conv_out, x):
    batch_size = x.shape[0]
    channels = x.shape[1]
    height = x.shape[2]
    width = x.shape[3]
    
    out = torch.empty((batch_size, channels), device=x.device, dtype=x.dtype)
    
    grid = lambda META: (batch_size * ((channels + META['BLOCK_C'] - 1) // META['BLOCK_C']),)
    
    fused_hardsigmoid_mul_avgpool_kernel[grid](
        conv_out_ptr=conv_out,
        x_ptr=x,
        out_ptr=out,
        batch_size=batch_size,
        channels=channels,
        height=height,
        width=width,
        conv_stride_b=conv_out.stride(0),
        conv_stride_c=conv_out.stride(1),
        x_stride_b=x.stride(0),
        x_stride_c=x.stride(1),
        x_stride_h=x.stride(2),
        x_stride_w=x.stride(3),
        out_stride_b=out.stride(0),
        out_stride_c=out.stride(1),
    )
    
    return out


def replacement_func():
    return fused_hardsigmoid_mul_avgpool