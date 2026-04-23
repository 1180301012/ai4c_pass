import torch
import triton
import triton.language as tl

# Pattern must mirror model.py exactly (positional args, no cleanup stmts)
def pattern(in_0, in_1, in_2, in_3):
    conv2d = torch.conv2d(in_3, in_1, in_0, (1, 1), (0, 0), (1, 1), 1)
    tmp_3 = conv2d.sigmoid()
    tmp_4 = in_2 * tmp_3
    tmp_5 = torch.nn.functional.hardtanh(tmp_4, 0.0, 6.0, False)
    return (tmp_5,)

def replacement_args(in_0, in_1, in_2, in_3):
    return (in_0, in_1, in_2, in_3)

# ─── Fully fused kernel: conv1x1 + sigmoid + broadcast multiply + hardtanh ──
# Each program handles one (b, c_out) pair:
#   1. Compute conv_val = dot(in_3[b, :], weight[c_out, :]) + bias[c_out]
#   2. Compute sig_val = sigmoid(conv_val)  (cast to fp32 for bf16/fp16)
#   3. For each spatial pos: output[b, c_out, h, w] = clamp(in_2[b, c_out, h, w] * sig_val, 0, 6)
# No intermediate tensors needed!

@triton.jit
def fused_se_kernel(
    in2_ptr, in3_ptr, weight_ptr, bias_ptr, output_ptr,
    in2_stride_b, in2_stride_c, in2_stride_h, in2_stride_w,
    in3_stride_b, in3_stride_c,
    weight_stride_oc, weight_stride_ic,
    Cout, Cin, H_dim, W_dim,
    BLOCK_CIN: tl.constexpr,
    BLOCK_HW: tl.constexpr,
):
    bc = tl.program_id(0)  # flattened (b, c_out) index
    b = bc // Cout
    c_out = bc % Cout

    # ── Step 1: Compute conv1x1 + sigmoid ──
    # Load bias (cast to fp32 for sigmoid computation)
    bias_val = tl.load(bias_ptr + c_out * weight_stride_oc).to(tl.float32)

    # Dot product: sum(in_3[b, ic] * weight[c_out, ic]) + bias
    ic_offsets = tl.arange(0, BLOCK_CIN)
    ic_mask = ic_offsets < Cin

    in3_offsets = b * in3_stride_b + ic_offsets * in3_stride_c
    in3_vals = tl.load(in3_ptr + in3_offsets, mask=ic_mask, other=0.0).to(tl.float32)

    w_offsets = c_out * weight_stride_oc + ic_offsets * weight_stride_ic
    w_vals = tl.load(weight_ptr + w_offsets, mask=ic_mask, other=0.0).to(tl.float32)

    conv_val = tl.sum(in3_vals * w_vals, axis=0) + bias_val

    # Sigmoid (always in fp32 for correctness across dtypes)
    sig_val = tl.sigmoid(conv_val)

    # ── Step 2: Broadcast multiply + hardtanh across spatial positions ──
    HW = H_dim * W_dim
    for hw_start in range(0, HW, BLOCK_HW):
        hw_offsets = hw_start + tl.arange(0, BLOCK_HW)
        hw_mask = hw_offsets < HW

        # Convert flat hw to (h, w) for stride-based indexing
        h_offsets = hw_offsets // W_dim
        w_spatial = hw_offsets % W_dim

        # Load feature map value using proper strides
        feat_offsets = b * in2_stride_b + c_out * in2_stride_c + h_offsets * in2_stride_h + w_spatial * in2_stride_w
        feat_val = tl.load(in2_ptr + feat_offsets, mask=hw_mask, other=0.0).to(tl.float32)

        # Multiply by sigmoid scale + hardtanh clamp [0, 6]
        out_val = tl.minimum(tl.maximum(feat_val * sig_val, 0.0), 6.0)

        # Store result (output is contiguous)
        out_offsets = b * Cout * HW + c_out * HW + hw_offsets
        tl.store(output_ptr + out_offsets, out_val, mask=hw_mask)


@torch.fx.wrap
def fused_conv1x1_sigmoid_mul_hardtanh(in_0, in_1, in_2, in_3):
    """
    Fully fused: conv2d(1x1) -> sigmoid -> broadcast multiply -> hardtanh(0, 6)
    No intermediate tensor allocations between operations.
    """
    Cout, Cin = in_1.shape[0], in_1.shape[1]
    B = in_3.shape[0]
    H_dim, W_dim = in_2.shape[2], in_2.shape[3]

    # Ensure weight and bias are on the same device as the feature map
    weight = in_1.to(in_2.device) if in_1.device != in_2.device else in_1
    bias = in_0.to(in_2.device) if in_0.device != in_2.device else in_0
    in3 = in_3.to(in_2.device) if in_3.device != in_2.device else in_3

    # Make in_2 contiguous for efficient memory access
    in2 = in_2.contiguous()

    BLOCK_CIN = triton.next_power_of_2(Cin)
    BLOCK_HW = 256  # Spatial tile size

    output = torch.empty((B, Cout, H_dim, W_dim), device=in_2.device, dtype=in_2.dtype)

    grid = (B * Cout,)

    fused_se_kernel[grid](
        in2_ptr=in2, in3_ptr=in3, weight_ptr=weight, bias_ptr=bias, output_ptr=output,
        in2_stride_b=in2.stride(0), in2_stride_c=in2.stride(1),
        in2_stride_h=in2.stride(2), in2_stride_w=in2.stride(3),
        in3_stride_b=in3.stride(0), in3_stride_c=in3.stride(1),
        weight_stride_oc=weight.stride(0), weight_stride_ic=max(weight.stride(1), 1),
        Cout=Cout, Cin=Cin, H_dim=H_dim, W_dim=W_dim,
        BLOCK_CIN=BLOCK_CIN,
        BLOCK_HW=BLOCK_HW,
    )

    return (output,)


def replacement_func():
    return fused_conv1x1_sigmoid_mul_hardtanh