import torch
import triton
import triton.language as tl


def pattern(in_0, in_1, in_2):
    """
    Match the pattern: sigmoid(interpolate(x)) * y
    Pattern to match:
    - sigmoid(conv2d)
    - interpolate to (64, 128)
    - multiply with in_2
    """
    # Note: Based on the model, this is: sigmoid -> interpolate -> multiply
    # We need to match the full pattern including the conv2d that feeds into sigmoid
    conv2d = torch.conv2d(in_1, in_0, None, (1, 1), (0, 0), (1, 1), 1)
    tmp_2 = torch.sigmoid(conv2d)
    tmp_3 = torch.nn.functional.interpolate(tmp_2, (64, 128), None, 'bilinear', False)
    tmp_4 = in_2 * tmp_3
    return tmp_4


def replacement_args(in_0, in_1, in_2):
    """
    Extract arguments needed for the replacement function.
    """
    return (in_0, in_1, in_2)


@triton.jit
def sigmoid_interpolate_mul_kernel(
    # Conv weight pointer
    weight_ptr,
    # Conv input pointer  
    input_ptr,
    # Multiplier tensor pointer
    mul_ptr,
    # Output pointer
    out_ptr,
    # Tensor dimensions
    in_c_out: tl.constexpr,
    in_c_in: tl.constexpr,
    in_h: tl.constexpr,
    in_w: tl.constexpr,
    out_h: tl.constexpr,
    out_w: tl.constexpr,
    # Conv parameters
    conv_kernel_h: tl.constexpr,
    conv_kernel_w: tl.constexpr,
    # Block sizes
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    """
    Fused kernel: Conv2D + Sigmoid + Interpolate + Mul
    
    Conv2D: (1, in_c_in, in_h, in_w) with (in_c_out, in_c_in, conv_kernel_h, conv_kernel_w) -> (1, in_c_out, in_h, in_w)
    Sigmoid: (1, in_c_out, in_h, in_w) -> (1, in_c_out, in_h, in_w)
    Interpolate: (1, in_c_out, in_h, in_w) -> (1, in_c_out, out_h, out_w)
    Mul: (1, in_c_out, out_h, out_w) * (1, in_c_out, out_h, out_w) -> (1, in_c_out, out_h, out_w)
    """
    # Program indices
    batch_idx = tl.program_id(0)
    out_c_idx = tl.program_id(1)
    
    # Output dimensions
    out_c = in_c_out
    out_spatial_size = out_h * out_w
    
    # Calculate output offsets for this program
    out_offset_base = batch_idx * out_c * out_spatial_size + out_c_idx * out_spatial_size
    out_row_offsets = out_offset_base + tl.arange(0, out_h)[:, None] * out_w + tl.arange(0, out_w)[None, :]
    out_mask = (out_row_offsets < (batch_idx + 1) * out_c * out_spatial_size)
    
    # Compute conv result for this output channel
    # Conv output at position (b, oc, oh, ow) = sum over ic, kh, kw of input[b, ic, oh+ph, ow+pw] * weight[oc, ic, kh, kw]
    # For padding=0, stride=1, dilation=1: oh = ih - kh, ow = iw - kw
    
    # The conv kernel is 1x1, so:
    # conv[b, oc, ih, iw] = sum over ic of input[b, ic, ih, iw] * weight[oc, ic, 0, 0]
    
    # We'll compute this using a reduction approach
    conv_result = tl.zeros((out_h, out_w), dtype=tl.float32)
    
    # Input dimensions for iteration
    in_spatial_size = in_h * in_w
    
    # Weight offset for this output channel
    weight_offset = out_c_idx * in_c_in * conv_kernel_h * conv_kernel_w
    
    # For each input channel, load weight and compute partial convolution
    for ic in range(in_c_in):
        # Load weight for this (oc, ic)
        w_offset = weight_offset + ic * conv_kernel_h * conv_kernel_w
        w = tl.load(weight_ptr + w_offset)  # weight[oc, ic, 0, 0]
        
        # Compute convolution contribution for this input channel
        # conv[b, oc, ih, iw] += input[b, ic, ih, iw] * weight[oc, ic, 0, 0]
        # We need to compute the interpolated output which requires knowing spatial relationships
        
        # Since this is 1x1 conv, the spatial output is at the same spatial position
        # For each spatial position in output, we accumulate over input channels
        pass  # Placeholder, we need a different approach
    
    # Let me use a cleaner approach with loop over output spatial positions
    # For each output spatial position (oh, ow), we need to compute:
    # conv[b, oc, oh, ow] = sum_ic(input[b, ic, oh, ow] * weight[oc, ic, 0, 0])
    
    # Actually, let's compute the conv result for all spatial positions first
    # Then apply sigmoid, interpolate, and multiply


@triton.jit
def sigmoid_interpolate_mul_kernel_v2(
    # Conv weight pointer
    weight_ptr,
    # Conv input pointer  
    input_ptr,
    # Multiplier tensor pointer
    mul_ptr,
    # Output pointer
    out_ptr,
    # Tensor dimensions
    batch: tl.constexpr,
    in_c_out: tl.constexpr,
    in_c_in: tl.constexpr,
    in_h: tl.constexpr,
    in_w: tl.constexpr,
    out_h: tl.constexpr,
    out_w: tl.constexpr,
    # Stride for input access
    in_spatial_size: tl.constexpr,
    out_spatial_size: tl.constexpr,
    total_out_elements: tl.constexpr,
    # Block size
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fused kernel: Conv2D (1x1) + Sigmoid + Interpolate + Mul
    
    Optimized for the specific case where:
    - Conv is 1x1 with stride=1, padding=0
    - Interpolation is bilinear upsampling
    """
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < total_out_elements
    
    # Each output element is at (b, oc, oh, ow)
    # Calculate which batch, channel, and spatial position this corresponds to
    b = offsets // (in_c_out * out_spatial_size)
    remainder = offsets % (in_c_out * out_spatial_size)
    oc = remainder // out_spatial_size
    oh_ow = remainder % out_spatial_size
    oh = oh_ow // out_w
    ow = oh_ow % out_w
    
    # For bilinear interpolation from (in_h, in_w) to (out_h, out_w):
    # oh_f = (oh + 0.5) * in_h / out_h - 0.5
    # ow_f = (ow + 0.5) * in_w / out_w - 0.5
    # Then sample 4 neighboring pixels and interpolate
    
    scale_h = tl.cast(in_h, tl.float32) / tl.cast(out_h, tl.float32)
    scale_w = tl.cast(in_w, tl.float32) / tl.cast(out_w, tl.float32)
    
    ih_f = (tl.cast(oh, tl.float32) + 0.5) * scale_h - 0.5
    iw_f = (tl.cast(ow, tl.float32) + 0.5) * scale_w - 0.5
    
    # Clamp to valid range
    ih_f = tl.maximum(tl.minimum(ih_f, tl.cast(in_h - 1, tl.float32)), 0.0)
    iw_f = tl.maximum(tl.minimum(iw_f, tl.cast(in_w - 1, tl.float32)), 0.0)
    
    # Get the 4 nearest integer coordinates
    ih_0 = tl.cast(tl.floor(ih_f), tl.int32)
    iw_0 = tl.cast(tl.floor(iw_f), tl.int32)
    ih_1 = tl.minimum(ih_0 + 1, in_h - 1)
    iw_1 = tl.minimum(iw_0 + 1, in_w - 1)
    
    # Interpolation weights
    ih_frac = ih_f - tl.cast(ih_0, tl.float32)
    iw_frac = iw_f - tl.cast(iw_0, tl.float32)
    
    # Compute the 1x1 conv result at the 4 interpolation points
    # For 1x1 conv: conv[b, oc, ih, iw] = sum_ic(input[b, ic, ih, iw] * weight[oc, ic, 0, 0])
    
    # We'll use a loop to accumulate over input channels
    conv_vals = tl.zeros((4,), dtype=tl.float32)  # Store conv values at 4 points
    
    for ic in range(in_c_in):
        # Input offset for batch b, channel ic
        in_base = b * in_c_in * in_spatial_size + ic * in_spatial_size
        
        # Load input at 4 neighboring positions
        # Note: For 1x1 conv, spatial position is preserved
        in_0 = tl.load(input_ptr + in_base + ih_0 * in_w + iw_0)
        in_1 = tl.load(input_ptr + in_base + ih_0 * in_w + iw_1)
        in_2 = tl.load(input_ptr + in_base + ih_1 * in_w + iw_0)
        in_3 = tl.load(input_ptr + in_base + ih_1 * in_w + iw_1)
        
        # Weight offset for output channel oc, input channel ic
        w_offset = oc * in_c_in + ic
        
        # Compute conv contribution at 4 points
        w = tl.load(weight_ptr + w_offset)
        conv_vals = conv_vals + tl.stack([
            in_0 * w,
            in_1 * w,
            in_2 * w,
            in_3 * w
        ])
    
    # Apply sigmoid to conv values
    sig_0 = 1.0 / (1.0 + tl.exp(-conv_vals[0]))
    sig_1 = 1.0 / (1.0 + tl.exp(-conv_vals[1]))
    sig_2 = 1.0 / (1.0 + tl.exp(-conv_vals[2]))
    sig_3 = 1.0 / (1.0 + tl.exp(-conv_vals[3]))
    
    # Bilinear interpolation of sigmoid values
    sig_h0 = sig_0 * (1.0 - ih_frac) + sig_2 * ih_frac
    sig_h1 = sig_1 * (1.0 - ih_frac) + sig_3 * ih_frac
    sig_interp = sig_h0 * (1.0 - iw_frac) + sig_h1 * iw_frac
    
    # Multiply with input tensor mul_ptr
    # mul tensor is at (b, oc, oh, ow)
    mul_offset = b * in_c_out * out_spatial_size + oc * out_spatial_size + oh_ow
    mul_val = tl.load(mul_ptr + mul_offset)
    
    # Final result
    result = sig_interp * mul_val
    
    # Store result
    out_offset = b * in_c_out * out_spatial_size + oc * out_spatial_size + oh_ow
    tl.store(out_ptr + offsets, result, mask=mask)


@torch.fx.wrap
def sigmoid_interpolate_mul_wrapper(in_0, in_1, in_2):
    """
    Wrapper function for the fused sigmoid + interpolate + multiply kernel.
    
    Args:
        in_0: Conv weight tensor (in_c_out, in_c_in, 1, 1)
        in_1: Conv input tensor (batch, in_c_in, in_h, in_w)
        in_2: Multiplier tensor (batch, in_c_out, out_h, out_w)
    
    Returns:
        Output tensor after Conv2D + Sigmoid + Interpolate + Multiply
    """
    # Get dimensions
    batch, in_c_in, in_h, in_w = in_1.shape
    in_c_out, _, conv_h, conv_w = in_0.shape
    _, _, out_h, out_w = in_2.shape
    
    out_spatial_size = out_h * out_w
    total_out_elements = batch * in_c_out * out_spatial_size
    
    # Allocate output
    out = torch.empty((batch, in_c_out, out_h, out_w), 
                     dtype=in_1.dtype, device=in_1.device)
    
    # Configure block size
    BLOCK_SIZE = 128
    num_programs = (total_out_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Launch kernel
    sigmoid_interpolate_mul_kernel_v2[(num_programs,)](
        weight_ptr=in_0,
        input_ptr=in_1,
        mul_ptr=in_2,
        out_ptr=out,
        batch=batch,
        in_c_out=in_c_out,
        in_c_in=in_c_in,
        in_h=in_h,
        in_w=in_w,
        out_h=out_h,
        out_w=out_w,
        in_spatial_size=in_h * in_w,
        out_spatial_size=out_spatial_size,
        total_out_elements=total_out_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out


def replacement_func():
    """
    Return the replacement function.
    """
    return sigmoid_interpolate_mul_wrapper