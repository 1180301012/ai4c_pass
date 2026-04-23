import torch
import triton
import triton.language as tl
import math

# Pattern matching function - must mirror the model.py exactly
def pattern(in_0: torch.Tensor, in_1, in_2, in_3):
    conv2d = torch.conv2d(in_3, in_1, in_0, (1, 1), (0, 0), (1, 1), 1)
    tmp_3 = conv2d.sigmoid()
    tmp_4 = in_2 * tmp_3
    tmp_5 = torch.nn.functional.gelu(tmp_4, approximate='none')
    tmp_6 = torch.nn.functional.adaptive_avg_pool2d(tmp_5, 1)
    tmp_7 = tmp_6.flatten(1, -1)
    tmp_8 = torch.nn.functional.dropout(tmp_7, 0.0, False, False)
    return (tmp_8,)

# Argument extraction
def replacement_args(in_0: torch.Tensor, in_1, in_2, in_3):
    return (in_0, in_1, in_2, in_3)

# Fused Triton kernel for the SE block
# Computes: conv2d -> sigmoid -> mul -> gelu -> avg_pool -> flatten -> identity_dropout
# All in one pass, avoiding intermediate tensor allocations
@triton.jit
def se_block_kernel(
    # Pointers
    feat_ptr,       # in_2: [N, C_out, H, W] - feature map
    conv_weight_ptr, # in_1: [C_out, C_in, 1, 1] - conv weight
    conv_bias_ptr,   # in_0: [C_out] - conv bias
    conv_input_ptr,  # in_3: [N, C_in, h_in, w_in] - conv input
    out_ptr,         # output: [N, C_out] - final result after pool+flatten
    # Dimensions
    N, C_out, C_in, H, W,
    # Strides for feat (in_2): [N, C_out, H, W]
    feat_stride_n, feat_stride_c, feat_stride_h, feat_stride_w,
    # Strides for conv_input (in_3): [N, C_in, h_in, w_in]
    cin_stride_n, cin_stride_c, cin_stride_h, cin_stride_w,
    # Strides for conv_weight: [C_out, C_in, 1, 1]
    cw_stride_o, cw_stride_i, cw_stride_h, cw_stride_w,
    # Strides for conv_bias: [C_out]
    cb_stride,
    # Strides for output: [N, C_out]
    out_stride_n, out_stride_c,
    # Block sizes
    BLOCK_C_OUT: tl.constexpr,
    BLOCK_C_IN: tl.constexpr,
    BLOCK_HW: tl.constexpr,
):
    # Program ID determines which (batch, channel_out_block) we handle
    pid_n = tl.program_id(0)
    pid_c_block = tl.program_id(1)

    # Channel output range for this program
    c_out_start = pid_c_block * BLOCK_C_OUT
    c_out_offsets = c_out_start + tl.arange(0, BLOCK_C_OUT)
    c_out_mask = c_out_offsets < C_out

    # Load conv bias for this block of output channels - cast to float32 for computation
    bias = tl.load(conv_bias_ptr + c_out_offsets * cb_stride, mask=c_out_mask, other=0.0).to(tl.float32)

    # Compute conv2d for this batch element and block of output channels
    # conv2d(input, weight, bias) = sum over C_in of weight * input + bias
    c_in_offsets = tl.arange(0, BLOCK_C_IN)
    c_in_mask = c_in_offsets < C_in

    # Load conv input - cast to float32
    conv_in = tl.load(conv_input_ptr + pid_n * cin_stride_n + c_in_offsets * cin_stride_c, mask=c_in_mask, other=0.0).to(tl.float32)

    # Load conv weight - cast to float32
    weight = tl.load(conv_weight_ptr + c_out_offsets[:, None] * cw_stride_o + c_in_offsets[None, :] * cw_stride_i,
                     mask=c_out_mask[:, None] & c_in_mask[None, :], other=0.0).to(tl.float32)

    # conv result = sum(weight * input) + bias, shape [BLOCK_C_OUT] in float32
    conv_result = tl.sum(weight * conv_in[None, :], axis=1) + bias

    # sigmoid - works in float32
    sigmoid_result = tl.sigmoid(conv_result)  # [BLOCK_C_OUT], float32

    # Accumulator for avg pool in float32
    acc = tl.zeros([BLOCK_C_OUT, BLOCK_HW], dtype=tl.float32)

    # Iterate over spatial dimensions
    hw_total = H * W
    num_hw_blocks = tl.cdiv(hw_total, BLOCK_HW)

    for hw_block_idx in range(num_hw_blocks):
        hw_start = hw_block_idx * BLOCK_HW
        hw_offsets = hw_start + tl.arange(0, BLOCK_HW)
        hw_mask = hw_offsets < hw_total

        # Convert flat HW offset to h, w coordinates
        h_coords = hw_offsets // W
        w_coords = hw_offsets % W

        # Load feature map: feat[n, c_out, h, w] - cast to float32
        feat = tl.load(feat_ptr + pid_n * feat_stride_n +
                        c_out_offsets[:, None] * feat_stride_c +
                        h_coords[None, :] * feat_stride_h +
                        w_coords[None, :] * feat_stride_w,
                        mask=c_out_mask[:, None] & hw_mask[None, :], other=0.0).to(tl.float32)

        # Multiply by sigmoid scale (broadcast over spatial)
        scaled = feat * sigmoid_result[:, None]

        # GELU: x * 0.5 * (1 + erf(x / sqrt(2)))
        gelu_result = scaled * 0.5 * (1.0 + tl.erf(scaled / 1.4142135623730951))

        # Accumulate for average pool
        acc += tl.where(c_out_mask[:, None] & hw_mask[None, :], gelu_result, 0.0)

    # Average over spatial dimensions
    avg = tl.sum(acc, axis=1) / hw_total  # [BLOCK_C_OUT], float32

    # Store output: [N, C_out] (after flatten from [N, C_out, 1, 1])
    tl.store(out_ptr + pid_n * out_stride_n + c_out_offsets * out_stride_c,
             avg, mask=c_out_mask)


@torch.fx.wrap
def fused_se_block(in_0, in_1, in_2, in_3):
    """
    Fused implementation of:
    conv2d(in_3, in_1, in_0) -> sigmoid -> mul(in_2) -> gelu -> avg_pool(1) -> flatten -> dropout(0)
    """
    # in_0: bias [C_out]
    # in_1: weight [C_out, C_in, 1, 1]
    # in_2: feature map [N, C_out, H, W]
    # in_3: conv input [N, C_in, h_in, w_in]

    N = in_2.shape[0]
    C_out = in_2.shape[1]
    H = in_2.shape[2]
    W = in_2.shape[3]
    C_in = in_1.shape[1]

    hw_total = H * W

    # Output: [N, C_out] (after avg_pool -> flatten -> identity dropout)
    # Output dtype matches input dtype
    out = torch.empty((N, C_out), dtype=in_2.dtype, device=in_2.device)

    # Choose block sizes based on dimensions
    BLOCK_C_OUT = min(64, triton.next_power_of_2(C_out))
    BLOCK_C_IN = min(64, triton.next_power_of_2(C_in))
    BLOCK_HW = 256

    grid = (N, triton.cdiv(C_out, BLOCK_C_OUT))
    se_block_kernel[grid](
        feat_ptr=in_2,
        conv_weight_ptr=in_1,
        conv_bias_ptr=in_0,
        conv_input_ptr=in_3,
        out_ptr=out,
        N=N, C_out=C_out, C_in=C_in, H=H, W=W,
        feat_stride_n=in_2.stride(0), feat_stride_c=in_2.stride(1),
        feat_stride_h=in_2.stride(2), feat_stride_w=in_2.stride(3),
        cin_stride_n=in_3.stride(0), cin_stride_c=in_3.stride(1),
        cin_stride_h=in_3.stride(2), cin_stride_w=in_3.stride(3),
        cw_stride_o=in_1.stride(0), cw_stride_i=in_1.stride(1),
        cw_stride_h=in_1.stride(2), cw_stride_w=in_1.stride(3),
        cb_stride=in_0.stride(0),
        out_stride_n=out.stride(0), out_stride_c=out.stride(1),
        BLOCK_C_OUT=BLOCK_C_OUT,
        BLOCK_C_IN=BLOCK_C_IN,
        BLOCK_HW=BLOCK_HW,
    )

    return (out,)

def replacement_func():
    return fused_se_block