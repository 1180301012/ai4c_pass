import torch
import triton
import triton.language as tl

def pattern(in_0, in_1, in_2):
    """
    Pattern to match: Conv2D followed by concatenation
    in_0: weight tensor [out_c, in_c, 3, 3]
    in_1: input tensor [batch, in_c, h, w]
    in_2: tensor to concatenate [batch, other_c, h, w]
    """
    tmp_1 = torch.conv2d(in_1, in_0, None, (1, 1), (1, 1), (1, 1), 1)
    tmp_2 = torch.cat((tmp_1, in_2), 1)
    return tmp_2

def replacement_args(in_0, in_1, in_2):
    return (in_0, in_1, in_2)

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 128, 'BLOCK_OC': 4}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 256, 'BLOCK_OC': 4}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 512, 'BLOCK_OC': 8}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 1024, 'BLOCK_OC': 8}, num_warps=8),
    ],
    key=['total_spatial', 'in_channels', 'out_channels'],
)
@triton.jit
def conv2d_3x3_kernel(
    input_ptr, weight_ptr, output_ptr,
    batch, in_channels, out_channels, height, width,
    input_b_stride, input_c_stride, input_h_stride, input_w_stride,
    weight_oc_stride, weight_ic_stride, weight_kh_stride, weight_kw_stride,
    output_b_stride, output_c_stride, output_h_stride, output_w_stride,
    total_spatial,
    BLOCK_SIZE: tl.constexpr, BLOCK_OC: tl.constexpr,
):
    # Process BLOCK_SIZE spatial positions and BLOCK_OC output channels per block
    pid_spatial = tl.program_id(0)
    pid_oc_group = tl.program_id(1)
    
    # Calculate spatial offsets for this block
    spatial_offsets = pid_spatial * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    spatial_mask = spatial_offsets < total_spatial
    
    # Decompose spatial offsets into (batch, h, w)
    b = spatial_offsets // (height * width)
    remainder = spatial_offsets % (height * width)
    h = remainder // width
    w = remainder % width
    
    # Calculate output channel range for this block
    oc_start = pid_oc_group * BLOCK_OC
    oc_offsets = oc_start + tl.arange(0, BLOCK_OC)
    oc_mask = oc_offsets < out_channels
    
    # Initialize accumulators for all output channels
    acc = tl.zeros((BLOCK_OC, BLOCK_SIZE), dtype=tl.float32)
    
    # Loop over input channels and 3x3 kernel
    for ic in range(in_channels):
        # Load all 9 weight values for this input channel
        for kh in range(3):
            for kw in range(3):
                # Load weights for all output channels
                weight_offsets = (oc_offsets[:, None] * weight_oc_stride +
                                ic * weight_ic_stride +
                                kh * weight_kh_stride +
                                kw * weight_kw_stride)
                weights = tl.load(weight_ptr + weight_offsets, mask=oc_mask[:, None], other=0.0)
                
                # Calculate input positions with padding
                ih = h + kh - 1
                iw = w + kw - 1
                
                # Create mask for valid input positions
                valid_mask = (ih >= 0) & (ih < height) & (iw >= 0) & (iw < width) & spatial_mask
                
                # Load input values
                input_offsets = (b * input_b_stride +
                               ic * input_c_stride +
                               ih * input_h_stride +
                               iw * input_w_stride)
                input_vals = tl.load(input_ptr + input_offsets, mask=valid_mask, other=0.0)
                
                # Accumulate: broadcast input across output channels
                acc += weights * input_vals[None, :]
    
    # Store results
    output_offsets = (b[None, :] * output_b_stride +
                     oc_offsets[:, None] * output_c_stride +
                     h[None, :] * output_h_stride +
                     w[None, :] * output_w_stride)
    store_mask = spatial_mask[None, :] & oc_mask[:, None]
    tl.store(output_ptr + output_offsets, acc, mask=store_mask)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 512}, num_warps=2),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8),
    ],
    key=['total_elements'],
)
@triton.jit
def copy_kernel(
    src_ptr, dst_ptr, total_elements,
    src_b_stride, src_c_stride, src_h_stride, src_w_stride,
    dst_b_stride, dst_c_stride, dst_h_stride, dst_w_stride,
    batch, channels, height, width,
    c_offset,  # channel offset in destination
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    
    # Calculate offsets
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < total_elements
    
    # Decompose linear offset
    b = offsets // (channels * height * width)
    remainder = offsets % (channels * height * width)
    c = remainder // (height * width)
    remainder = remainder % (height * width)
    h = remainder // width
    w = remainder % width
    
    # Calculate source and destination offsets
    src_offset = (b * src_b_stride + c * src_c_stride + 
                  h * src_h_stride + w * src_w_stride)
    dst_offset = (b * dst_b_stride + (c + c_offset) * dst_c_stride + 
                  h * dst_h_stride + w * dst_w_stride)
    
    # Load and store
    data = tl.load(src_ptr + src_offset, mask=mask, other=0.0)
    tl.store(dst_ptr + dst_offset, data, mask=mask)


@torch.fx.wrap
def fused_conv2d_cat(weight, input, other):
    """
    Fused implementation of conv2d + cat
    """
    # Get dimensions
    batch, in_channels, height, width = input.shape
    out_channels = weight.shape[0]
    other_channels = other.shape[1]
    total_out_channels = out_channels + other_channels
    total_spatial = batch * height * width
    
    # Allocate output
    output = torch.empty(batch, total_out_channels, height, width,
                        device=input.device, dtype=input.dtype)
    
    # Launch conv2d kernel
    def grid_conv(meta):
        return (
            triton.cdiv(total_spatial, meta['BLOCK_SIZE']),
            triton.cdiv(out_channels, meta['BLOCK_OC'])
        )
    
    conv2d_3x3_kernel[grid_conv](
        input, weight, output,
        batch, in_channels, out_channels, height, width,
        input.stride(0), input.stride(1), input.stride(2), input.stride(3),
        weight.stride(0), weight.stride(1), weight.stride(2), weight.stride(3),
        output.stride(0), output.stride(1), output.stride(2), output.stride(3),
        total_spatial,
    )
    
    # Launch copy kernel for concatenation
    total_copy_elements = batch * other_channels * height * width
    grid_copy = lambda meta: (triton.cdiv(total_copy_elements, meta['BLOCK_SIZE']),)
    
    copy_kernel[grid_copy](
        other, output, total_copy_elements,
        other.stride(0), other.stride(1), other.stride(2), other.stride(3),
        output.stride(0), output.stride(1), output.stride(2), output.stride(3),
        batch, other_channels, height, width,
        out_channels,  # channel offset
    )
    
    return output

def replacement_func():
    return fused_conv2d_cat