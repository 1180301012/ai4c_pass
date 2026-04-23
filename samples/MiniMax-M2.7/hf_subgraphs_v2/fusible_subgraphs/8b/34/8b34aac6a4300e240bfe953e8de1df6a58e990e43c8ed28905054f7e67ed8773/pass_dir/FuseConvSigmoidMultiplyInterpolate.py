import torch
import triton
import triton.language as tl


@triton.jit
def fused_conv_sigmoid_multiply_interpolate_kernel(
    input_ptr,
    weight_ptr,
    bias_ptr,
    other_ptr,
    output_ptr,
    batch_size,
    in_channels,
    out_channels,
    in_height,
    in_width,
    out_height,
    out_width,
    num_kernel_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fused kernel: conv2d + sigmoid + multiply + interpolate (bilinear)
    
    conv2d with 1x1 kernel, stride=1, padding=0, dilation=1, groups=1
    """
    # Each program handles a portion of the output
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < num_kernel_elements
    
    # Calculate output indices: flatten (b, oc, oh, ow)
    tmp = offsets // (out_height * out_width)
    oc = tmp % out_channels
    b = tmp // out_channels
    tmp = offsets % (out_height * out_width)
    oh = tmp // out_width
    ow = tmp % out_width
    
    # Conv2d 1x1: output[b, oc, oh, ow] = sum_ic input[b, ic, oh, ow] * weight[oc, ic, 0, 0] + bias[oc]
    # Since it's 1x1 conv with stride=1, the spatial coordinates don't change
    conv_val = tl.load(bias_ptr + oc, mask=mask, other=0.0)
    
    # Accumulate over input channels
    for ic in range(in_channels):
        input_idx = b * in_channels * in_height * in_width + ic * in_height * in_width + oh * in_width + ow
        weight_idx = oc * in_channels + ic
        input_val = tl.load(input_ptr + input_idx, mask=mask, other=0.0)
        weight_val = tl.load(weight_ptr + weight_idx, mask=mask, other=0.0)
        conv_val = conv_val + input_val * weight_val
    
    # Apply sigmoid
    exp_neg = tl.exp(-conv_val)
    sigmoid_val = 1.0 / (1.0 + exp_neg)
    
    # Multiply with other
    other_idx = b * out_channels * out_height * out_width + oc * out_height * out_width + oh * out_width + ow
    other_val = tl.load(other_ptr + other_idx, mask=mask, other=0.0)
    mul_val = sigmoid_val * other_val
    
    # For interpolation, we need to map output position (oh, ow) to input position
    # The input to interpolate is mul_val (already computed at position oh, ow in the original input space)
    # Since we want to interpolate to out_height, out_width and oh, ow are the output coords in the 64x64 space,
    # we need to interpolate from the mul_val which is at position (oh, ow) in the 16x16 space
    
    # Actually, for this pattern: conv -> sigmoid -> multiply -> interpolate(to 64x64 from 16x16)
    # The interpolate takes mul_val at 16x16 and outputs at 64x64
    # But oh, ow are already in 64x64 space... so this kernel computes conv+sigmoid+multiply for 16x16,
    # then we'd still need interpolate. Let me reconsider.
    
    # Looking at the original pattern:
    # tmp_2 = conv2d(in_5, in_1, in_0, ...)  # output: 16x16
    # tmp_6 = sigmoid(tmp_2)  # 16x16
    # tmp_7 = in_2 * tmp_6  # 16x16
    # tmp_8 = interpolate(tmp_7, (64, 64), ...)  # 64x64
    
    # So the interpolate is on the result of multiply (16x16), and outputs 64x64
    # oh, ow are in 64x64 space, but mul_val is at (oh_scaled, ow_scaled) in 16x16 space
    
    # Calculate source position in 16x16 input space
    h_scale = (in_height - 1.0) / (out_height - 1.0) if out_height > 1 else 0.0
    w_scale = (in_width - 1.0) / (out_width - 1.0) if out_width > 1 else 0.0
    
    src_h = oh * h_scale
    src_w = ow * w_scale
    
    src_h0 = tl.floor(src_h)
    src_w0 = tl.floor(src_w)
    src_h1 = src_h0 + 1
    src_w1 = src_w0 + 1
    
    # Clamp
    src_h0 = tl.clamp(src_h0, 0, in_height - 1)
    src_w0 = tl.clamp(src_w0, 0, in_width - 1)
    src_h1 = tl.clamp(src_h1, 0, in_height - 1)
    src_w1 = tl.clamp(src_w1, 0, in_width - 1)
    
    # Interpolation weights
    h_l = src_h - src_h0
    w_l = src_w - src_w0
    h_h = 1.0 - h_l
    w_h = 1.0 - w_l
    
    # Load the conv+sigmoid+multiply result at the 4 neighboring positions
    # Position (src_h0, src_w0) in the 16x16 space
    # But wait, we need to compute conv+sigmoid+multiply at src_h0, src_w0 first!
    
    # This is complex because we need conv+sigmoid+multiply at different positions
    # Let me recompute for each interpolation corner
    
    # Top-left corner (src_h0, src_w0)
    conv_val_00 = tl.load(bias_ptr + oc, mask=mask, other=0.0)
    for ic in range(in_channels):
        input_idx = b * in_channels * in_height * in_width + ic * in_height * in_width + src_h0 * in_width + src_w0
        weight_idx = oc * in_channels + ic
        input_val = tl.load(input_ptr + input_idx, mask=mask, other=0.0)
        weight_val = tl.load(weight_ptr + weight_idx, mask=mask, other=0.0)
        conv_val_00 = conv_val_00 + input_val * weight_val
    
    exp_neg_00 = tl.exp(-conv_val_00)
    sigmoid_00 = 1.0 / (1.0 + exp_neg_00)
    other_idx_00 = b * out_channels * out_height * out_width + oc * out_height * out_width + src_h0 * out_width + src_w0
    other_00 = tl.load(other_ptr + other_idx_00, mask=mask, other=0.0)
    mul_00 = sigmoid_00 * other_00
    
    # Top-right corner (src_h0, src_w1)
    conv_val_01 = tl.load(bias_ptr + oc, mask=mask, other=0.0)
    for ic in range(in_channels):
        input_idx = b * in_channels * in_height * in_width + ic * in_height * in_width + src_h0 * in_width + src_w1
        weight_idx = oc * in_channels + ic
        input_val = tl.load(input_ptr + input_idx, mask=mask, other=0.0)
        weight_val = tl.load(weight_ptr + weight_idx, mask=mask, other=0.0)
        conv_val_01 = conv_val_01 + input_val * weight_val
    
    exp_neg_01 = tl.exp(-conv_val_01)
    sigmoid_01 = 1.0 / (1.0 + exp_neg_01)
    other_idx_01 = b * out_channels * out_height * out_width + oc * out_height * out_width + src_h0 * out_width + src_w1
    other_01 = tl.load(other_ptr + other_idx_01, mask=mask, other=0.0)
    mul_01 = sigmoid_01 * other_01
    
    # Bottom-left corner (src_h1, src_w0)
    conv_val_10 = tl.load(bias_ptr + oc, mask=mask, other=0.0)
    for ic in range(in_channels):
        input_idx = b * in_channels * in_height * in_width + ic * in_height * in_width + src_h1 * in_width + src_w0
        weight_idx = oc * in_channels + ic
        input_val = tl.load(input_ptr + input_idx, mask=mask, other=0.0)
        weight_val = tl.load(weight_ptr + weight_idx, mask=mask, other=0.0)
        conv_val_10 = conv_val_10 + input_val * weight_val
    
    exp_neg_10 = tl.exp(-conv_val_10)
    sigmoid_10 = 1.0 / (1.0 + exp_neg_10)
    other_idx_10 = b * out_channels * out_height * out_width + oc * out_height * out_width + src_h1 * out_width + src_w0
    other_10 = tl.load(other_ptr + other_idx_10, mask=mask, other=0.0)
    mul_10 = sigmoid_10 * other_10
    
    # Bottom-right corner (src_h1, src_w1)
    conv_val_11 = tl.load(bias_ptr + oc, mask=mask, other=0.0)
    for ic in range(in_channels):
        input_idx = b * in_channels * in_height * in_width + ic * in_height * in_width + src_h1 * in_width + src_w1
        weight_idx = oc * in_channels + ic
        input_val = tl.load(input_ptr + input_idx, mask=mask, other=0.0)
        weight_val = tl.load(weight_ptr + weight_idx, mask=mask, other=0.0)
        conv_val_11 = conv_val_11 + input_val * weight_val
    
    exp_neg_11 = tl.exp(-conv_val_11)
    sigmoid_11 = 1.0 / (1.0 + exp_neg_11)
    other_idx_11 = b * out_channels * out_height * out_width + oc * out_height * out_width + src_h1 * out_width + src_w1
    other_11 = tl.load(other_ptr + other_idx_11, mask=mask, other=0.0)
    mul_11 = sigmoid_11 * other_11
    
    # Bilinear interpolation
    v0 = h_h * mul_00 + h_l * mul_10
    v1 = h_h * mul_01 + h_l * mul_11
    result = w_h * v0 + w_l * v1
    
    # Store result
    out_idx = b * out_channels * out_height * out_width + oc * out_height * out_width + oh * out_width + ow
    tl.store(output_ptr + out_idx, result, mask=mask)


@torch.fx.wrap
def fused_conv_sigmoid_multiply_interpolate_wrapper(
    input_tensor, weight_tensor, bias_tensor, other_tensor, output_size
):
    """
    Fused conv2d + sigmoid + multiply + interpolate operation.
    
    input_tensor: [B, IC, H, W] - input to conv2d
    weight_tensor: [OC, IC, 1, 1] - conv2d weight
    bias_tensor: [OC] - conv2d bias
    other_tensor: [B, OC, H, W] - tensor to multiply with sigmoid(conv2d result)
    output_size: tuple (H_out, W_out)
    """
    batch_size, in_channels, in_height, in_width = input_tensor.shape
    out_channels = weight_tensor.shape[0]
    out_height, out_width = output_size
    
    # Allocate output
    output = torch.empty(
        (batch_size, out_channels, out_height, out_width),
        device=input_tensor.device,
        dtype=input_tensor.dtype
    )
    
    total_elements = batch_size * out_channels * out_height * out_width
    BLOCK_SIZE = 256  # Reduced for this complex kernel
    num_programs = (total_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    fused_conv_sigmoid_multiply_interpolate_kernel[(num_programs,)](
        input_tensor,
        weight_tensor,
        bias_tensor,
        other_tensor,
        output,
        batch_size,
        in_channels,
        out_channels,
        in_height,
        in_width,
        out_height,
        out_width,
        total_elements,
        BLOCK_SIZE
    )
    
    return output


def pattern(in_5, in_1, in_0, in_2):
    """
    Match the pattern: conv2d(in_5, in_1, in_0) → sigmoid → in_2 * sigmoid → interpolate(64, 64)
    Returns the result tensor.
    """
    tmp_0 = in_0
    tmp_1 = in_1
    tmp_2 = torch.conv2d(in_5, tmp_1, tmp_0, (1, 1), (0, 0), (1, 1), 1)
    tmp_6 = torch.sigmoid(tmp_2)
    tmp_7 = in_2 * tmp_6
    tmp_8 = torch.nn.functional.interpolate(tmp_7, (64, 64), None, 'bilinear', False)
    return tmp_8


def replacement_args(in_5, in_1, in_0, in_2):
    return (in_5, in_1, in_0, in_2)


def replacement_func():
    return fused_conv_sigmoid_multiply_interpolate_wrapper