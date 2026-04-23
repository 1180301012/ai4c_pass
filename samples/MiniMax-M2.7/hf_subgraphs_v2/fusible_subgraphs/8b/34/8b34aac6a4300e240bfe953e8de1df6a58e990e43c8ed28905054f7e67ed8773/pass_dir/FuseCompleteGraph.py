import torch
import triton
import triton.language as tl


@triton.jit
def complete_fusion_kernel(
    # Branch A: conv2d input, weight, bias
    branch_a_input_ptr,
    branch_a_weight_ptr,
    branch_a_bias_ptr,
    # Branch A: multiply with in_2
    branch_a_mult_ptr,
    # Branch B: interpolate input
    branch_b_input_ptr,
    # Branch B: multiply with in_3
    branch_b_mult_ptr,
    # Output
    output_ptr,
    # Dimensions
    batch_size,
    num_channels,
    branch_a_in_h,
    branch_a_in_w,
    out_h,
    out_w,
    # Grid info
    total_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Complete fusion kernel for BiSeNetV2 BGA module's semantic branch.
    
    Branch A: conv2d(input, weight, bias) -> sigmoid -> mult -> interpolate(to 64x64)
    Branch B: interpolate(input, to 64x64) -> sigmoid -> mult
    Output: branch_a_result + branch_b_result
    """
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < total_elements
    
    # Calculate output indices: flatten (b, c, oh, ow)
    tmp = offsets // (out_h * out_w)
    b = tmp % batch_size
    c = tmp // batch_size
    tmp = offsets % (out_h * out_w)
    oh = tmp // out_w
    ow = tmp % out_w
    
    # ============= Branch A: conv2d + sigmoid + multiply + interpolate =============
    # Conv2d 1x1: for position (oh, ow) in 16x16 space, compute conv
    # For interpolate from 16x16 to 64x64, we need bilinear interpolation
    
    # Calculate source position in 16x16 space
    h_scale = (branch_a_in_h - 1.0) / (out_h - 1.0) if out_h > 1 else 0.0
    w_scale = (branch_a_in_w - 1.0) / (out_w - 1.0) if out_w > 1 else 0.0
    
    src_h = oh * h_scale
    src_w = ow * w_scale
    
    src_h0 = tl.floor(src_h)
    src_w0 = tl.floor(src_w)
    src_h1 = src_h0 + 1
    src_w1 = src_w0 + 1
    
    # Clamp to valid range
    src_h0 = tl.clamp(src_h0, 0, branch_a_in_h - 1)
    src_w0 = tl.clamp(src_w0, 0, branch_a_in_w - 1)
    src_h1 = tl.clamp(src_h1, 0, branch_a_in_h - 1)
    src_w1 = tl.clamp(src_w1, 0, branch_a_in_w - 1)
    
    # Interpolation weights
    h_l = src_h - src_h0
    w_l = src_w - src_w0
    h_h = 1.0 - h_l
    w_h = 1.0 - w_l
    
    # For bilinear interpolation, we need conv+sigmoid+multiply at 4 corners
    # Top-left (src_h0, src_w0)
    conv_a_00 = tl.load(branch_a_bias_ptr + c, mask=mask, other=0.0)
    for ic in range(num_channels):
        input_idx = b * num_channels * branch_a_in_h * branch_a_in_w + ic * branch_a_in_h * branch_a_in_w + src_h0 * branch_a_in_w + src_w0
        weight_idx = c * num_channels + ic
        input_val = tl.load(branch_a_input_ptr + input_idx, mask=mask, other=0.0)
        weight_val = tl.load(branch_a_weight_ptr + weight_idx, mask=mask, other=0.0)
        conv_a_00 = conv_a_00 + input_val * weight_val
    
    sigmoid_a_00 = 1.0 / (1.0 + tl.exp(-conv_a_00))
    mult_a_idx_00 = b * num_channels * branch_a_in_h * branch_a_in_w + c * branch_a_in_h * branch_a_in_w + src_h0 * branch_a_in_w + src_w0
    mult_a_val_00 = tl.load(branch_a_mult_ptr + mult_a_idx_00, mask=mask, other=0.0)
    result_a_00 = sigmoid_a_00 * mult_a_val_00
    
    # Top-right (src_h0, src_w1)
    conv_a_01 = tl.load(branch_a_bias_ptr + c, mask=mask, other=0.0)
    for ic in range(num_channels):
        input_idx = b * num_channels * branch_a_in_h * branch_a_in_w + ic * branch_a_in_h * branch_a_in_w + src_h0 * branch_a_in_w + src_w1
        weight_idx = c * num_channels + ic
        input_val = tl.load(branch_a_input_ptr + input_idx, mask=mask, other=0.0)
        weight_val = tl.load(branch_a_weight_ptr + weight_idx, mask=mask, other=0.0)
        conv_a_01 = conv_a_01 + input_val * weight_val
    
    sigmoid_a_01 = 1.0 / (1.0 + tl.exp(-conv_a_01))
    mult_a_idx_01 = b * num_channels * branch_a_in_h * branch_a_in_w + c * branch_a_in_h * branch_a_in_w + src_h0 * branch_a_in_w + src_w1
    mult_a_val_01 = tl.load(branch_a_mult_ptr + mult_a_idx_01, mask=mask, other=0.0)
    result_a_01 = sigmoid_a_01 * mult_a_val_01
    
    # Bottom-left (src_h1, src_w0)
    conv_a_10 = tl.load(branch_a_bias_ptr + c, mask=mask, other=0.0)
    for ic in range(num_channels):
        input_idx = b * num_channels * branch_a_in_h * branch_a_in_w + ic * branch_a_in_h * branch_a_in_w + src_h1 * branch_a_in_w + src_w0
        weight_idx = c * num_channels + ic
        input_val = tl.load(branch_a_input_ptr + input_idx, mask=mask, other=0.0)
        weight_val = tl.load(branch_a_weight_ptr + weight_idx, mask=mask, other=0.0)
        conv_a_10 = conv_a_10 + input_val * weight_val
    
    sigmoid_a_10 = 1.0 / (1.0 + tl.exp(-conv_a_10))
    mult_a_idx_10 = b * num_channels * branch_a_in_h * branch_a_in_w + c * branch_a_in_h * branch_a_in_w + src_h1 * branch_a_in_w + src_w0
    mult_a_val_10 = tl.load(branch_a_mult_ptr + mult_a_idx_10, mask=mask, other=0.0)
    result_a_10 = sigmoid_a_10 * mult_a_val_10
    
    # Bottom-right (src_h1, src_w1)
    conv_a_11 = tl.load(branch_a_bias_ptr + c, mask=mask, other=0.0)
    for ic in range(num_channels):
        input_idx = b * num_channels * branch_a_in_h * branch_a_in_w + ic * branch_a_in_h * branch_a_in_w + src_h1 * branch_a_in_w + src_w0
        weight_idx = c * num_channels + ic
        input_val = tl.load(branch_a_input_ptr + input_idx, mask=mask, other=0.0)
        weight_val = tl.load(branch_a_weight_ptr + weight_idx, mask=mask, other=0.0)
        conv_a_11 = conv_a_11 + input_val * weight_val
    
    sigmoid_a_11 = 1.0 / (1.0 + tl.exp(-conv_a_11))
    mult_a_idx_11 = b * num_channels * branch_a_in_h * branch_a_in_w + c * branch_a_in_h * branch_a_in_w + src_h1 * branch_a_in_w + src_w1
    mult_a_val_11 = tl.load(branch_a_mult_ptr + mult_a_idx_11, mask=mask, other=0.0)
    result_a_11 = sigmoid_a_11 * mult_a_val_11
    
    # Bilinear interpolation for Branch A
    v0_a = h_h * result_a_00 + h_l * result_a_10
    v1_a = h_h * result_a_01 + h_l * result_a_11
    branch_a_result = w_h * v0_a + w_l * v1_a
    
    # ============= Branch B: interpolate + sigmoid + multiply =============
    # Branch B: in_4 (16x16) -> interpolate to 64x64 -> sigmoid -> multiply with in_3 (64x64)
    # in_4 spatial size is same as branch_a_in_h, branch_a_in_w (16x16)
    
    # Bilinear interpolation for Branch B input
    # h0, w0
    idx_h0_w0 = b * num_channels * branch_a_in_h * branch_a_in_w + c * branch_a_in_h * branch_a_in_w + src_h0 * branch_a_in_w + src_w0
    v_b_h0_w0 = tl.load(branch_b_input_ptr + idx_h0_w0, mask=mask, other=0.0)
    
    # h0, w1
    idx_h0_w1 = b * num_channels * branch_a_in_h * branch_a_in_w + c * branch_a_in_h * branch_a_in_w + src_h0 * branch_a_in_w + src_w1
    v_b_h0_w1 = tl.load(branch_b_input_ptr + idx_h0_w1, mask=mask, other=0.0)
    
    # h1, w0
    idx_h1_w0 = b * num_channels * branch_a_in_h * branch_a_in_w + c * branch_a_in_h * branch_a_in_w + src_h1 * branch_a_in_w + src_w0
    v_b_h1_w0 = tl.load(branch_b_input_ptr + idx_h1_w0, mask=mask, other=0.0)
    
    # h1, w1
    idx_h1_w1 = b * num_channels * branch_a_in_h * branch_a_in_w + c * branch_a_in_h * branch_a_in_w + src_h1 * branch_a_in_w + src_w1
    v_b_h1_w1 = tl.load(branch_b_input_ptr + idx_h1_w1, mask=mask, other=0.0)
    
    # Bilinear interpolation
    v0_b = h_h * v_b_h0_w0 + h_l * v_b_h1_w0
    v1_b = h_h * v_b_h0_w1 + h_l * v_b_h1_w1
    interpolated_b = w_h * v0_b + w_l * v1_b
    
    # Sigmoid
    sigmoid_b = 1.0 / (1.0 + tl.exp(-interpolated_b))
    
    # Multiply with in_3 (branch_b_mult)
    mult_b_idx = b * num_channels * out_h * out_w + c * out_h * out_w + oh * out_w + ow
    mult_b_val = tl.load(branch_b_mult_ptr + mult_b_idx, mask=mask, other=0.0)
    branch_b_result = sigmoid_b * mult_b_val
    
    # ============= Final: Add both branches =============
    final_result = branch_a_result + branch_b_result
    
    # Store result
    out_idx = b * num_channels * out_h * out_w + c * out_h * out_w + oh * out_w + ow
    tl.store(output_ptr + out_idx, final_result, mask=mask)


@torch.fx.wrap
def complete_fusion_wrapper(in_0, in_1, in_2, in_3, in_4, in_5):
    """
    Complete fusion of BiSeNetV2 BGA semantic branch.
    
    in_0: bias tensor [128]
    in_1: weight tensor [128, 128, 1, 1]
    in_2: multiply tensor for branch A [B, 128, 16, 16]
    in_3: multiply tensor for branch B [B, 128, 64, 64]
    in_4: interpolate input for branch B [B, 128, 16, 16]
    in_5: input tensor for branch A [B, 128, 16, 16]
    """
    bias = in_0
    weight = in_1
    branch_a_mult = in_2
    branch_b_mult = in_3
    branch_b_input = in_4
    branch_a_input = in_5
    
    batch_size, num_channels, in_h, in_w = branch_a_input.shape
    out_h = 64
    out_w = 64
    
    # Allocate output
    output = torch.empty(
        (batch_size, num_channels, out_h, out_w),
        device=branch_a_input.device,
        dtype=branch_a_input.dtype
    )
    
    total_elements = batch_size * num_channels * out_h * out_w
    BLOCK_SIZE = 64  # Small block size for this complex kernel with inner loops
    num_programs = (total_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    complete_fusion_kernel[(num_programs,)](
        branch_a_input,
        weight,
        bias,
        branch_a_mult,
        branch_b_input,
        branch_b_mult,
        output,
        batch_size,
        num_channels,
        in_h,
        in_w,
        out_h,
        out_w,
        total_elements,
        BLOCK_SIZE
    )
    
    return output


def pattern(in_0, in_1, in_2, in_3, in_4, in_5):
    """
    Match the complete BiSeNetV2 BGA semantic branch computation.
    Returns the final output tensor.
    """
    conv2d = torch.conv2d(in_5, in_1, in_0, (1, 1), (0, 0), (1, 1), 1)
    tmp_3 = torch.nn.functional.interpolate(in_4, (64, 64), None, 'bilinear', False)
    tmp_4 = torch.sigmoid(tmp_3)
    tmp_5 = in_3 * tmp_4
    tmp_6 = torch.sigmoid(conv2d)
    tmp_7 = in_2 * tmp_6
    tmp_8 = torch.nn.functional.interpolate(tmp_7, (64, 64), None, 'bilinear', False)
    tmp_9 = tmp_5 + tmp_8
    return tmp_9


def replacement_args(in_0, in_1, in_2, in_3, in_4, in_5):
    return (in_0, in_1, in_2, in_3, in_4, in_5)


def replacement_func():
    return complete_fusion_wrapper