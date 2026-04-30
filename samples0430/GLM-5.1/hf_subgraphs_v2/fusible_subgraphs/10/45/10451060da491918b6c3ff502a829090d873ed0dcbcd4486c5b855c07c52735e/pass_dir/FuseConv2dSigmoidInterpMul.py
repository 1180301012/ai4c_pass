import torch
import triton
import triton.language as tl


def pattern(in_0, in_1, in_2):
    conv2d = torch.conv2d(in_1, in_0, None, (1, 1), (0, 0), (1, 1), 1)
    tmp_2 = torch.sigmoid(conv2d)
    tmp_3 = torch.nn.functional.interpolate(tmp_2, (64, 128), None, 'bilinear', False)
    tmp_4 = in_2 * tmp_3
    return tmp_4


def replacement_args(in_0, in_1, in_2):
    return (in_0, in_1, in_2)


# ============================================================
# Kernel 1: 1x1 Conv2D + Sigmoid
# The 1x1 conv is equivalent to: weight[OC,IC] @ input[IC,IS]
# Produces conv_sig[OC, IS] in float32 for accuracy
# ============================================================
@triton.jit
def conv1x1_sigmoid_kernel(
    weight_ptr,
    input_ptr,
    output_ptr,
    OC: tl.constexpr,
    IC: tl.constexpr,
    IS: tl.constexpr,
    BLOCK_OC: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid = tl.program_id(0)
    oc = pid * BLOCK_OC + tl.arange(0, BLOCK_OC)
    oc_mask = oc < OC

    # Accumulator: [BLOCK_OC, IS] in float32
    acc = tl.zeros([BLOCK_OC, IS], dtype=tl.float32)

    # Loop over input channels in blocks of BLOCK_K
    for k_start in range(0, IC, BLOCK_K):
        k = k_start + tl.arange(0, BLOCK_K)
        k_mask = k < IC

        # Load weight block: [BLOCK_OC, BLOCK_K]
        w_offsets = oc[:, None] * IC + k[None, :]
        w_mask = oc_mask[:, None] & k_mask[None, :]
        w = tl.load(weight_ptr + w_offsets, mask=w_mask, other=0.0).to(tl.float32)

        # Load input block: [BLOCK_K, IS]
        i_offsets = k[:, None] * IS + tl.arange(0, IS)[None, :]
        i_mask = k_mask[:, None]
        inp = tl.load(input_ptr + i_offsets, mask=i_mask, other=0.0).to(tl.float32)

        # Matrix multiply accumulation
        acc += tl.dot(w, inp)

    # Apply sigmoid
    sig = tl.sigmoid(acc)

    # Store output: [BLOCK_OC, IS] in float32
    o_offsets = oc[:, None] * IS + tl.arange(0, IS)[None, :]
    o_mask = oc_mask[:, None]
    tl.store(output_ptr + o_offsets, sig, mask=o_mask)


# ============================================================
# Kernel 2: Bilinear Interpolate + Multiply
# Input: sig_out[OC, IS] (float32), in2[OC, OH, OW]
# Output: result[OC, OH, OW] = in2 * interp(sig_out)
#
# Since H_IN=1, height interpolation is trivial (always row 0).
# Width interpolation: src_w = (dst_w + 0.5) * (W_IN/W_OUT) - 0.5
# ============================================================
@triton.jit
def interp_mul_kernel(
    sig_out_ptr,
    in2_ptr,
    output_ptr,
    OC,
    OH,
    OW,
    IS,
    scale_w,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)

    total = OC * OH * OW
    mask = offsets < total

    # Decompose flat index to (c, h, w)
    w_out = offsets % OW
    c_out = offsets // (OH * OW)

    # Bilinear interpolation source coordinates (align_corners=False)
    # src_w = (dst_w + 0.5) * (W_IN / W_OUT) - 0.5
    w_src = (w_out.to(tl.float32) + 0.5) * scale_w - 0.5

    # Compute interpolation indices and fractional weight
    w0_raw = tl.math.floor(w_src)
    w1_raw = w0_raw + 1.0
    # Clamp indices independently to [0, IS-1]
    w0_int = tl.maximum(w0_raw.to(tl.int32), 0)
    w1_int = tl.minimum(w1_raw.to(tl.int32), IS - 1)
    # Fractional weight uses unclamped floor value
    w_frac = w_src - w0_raw

    # Load sigmoid output values at interpolation source positions
    # sig_out layout: [OC, IS], offset = c * IS + w
    sig_off0 = c_out * IS + w0_int
    sig_off1 = c_out * IS + w1_int
    val_w0 = tl.load(sig_out_ptr + sig_off0, mask=mask, other=0.0)
    val_w1 = tl.load(sig_out_ptr + sig_off1, mask=mask, other=0.0)

    # Width interpolation (height is trivial since H_IN=1, always uses row 0)
    interp_val = (1.0 - w_frac) * val_w0 + w_frac * val_w1

    # Load in_2 value and multiply
    in2_val = tl.load(in2_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    out_val = in2_val * interp_val

    # Store result
    tl.store(output_ptr + offsets, out_val, mask=mask)


# ============================================================
# Fused wrapper: computes conv2d+sigmoid then interpolate+multiply
# ============================================================
@torch.fx.wrap
def fused_conv_sigmoid_interp_mul(weight, input_feat, target):
    OC = weight.shape[0]   # 128
    IC = weight.shape[1]   # 960
    IS = input_feat.shape[3]  # 4
    OH = target.shape[2]   # 64
    OW = target.shape[3]   # 128

    # Ensure tensors are contiguous for Triton pointer arithmetic
    weight = weight.contiguous()
    input_feat = input_feat.contiguous()
    target = target.contiguous()

    # Step 1: Compute 1x1 conv2d + sigmoid -> small intermediate [OC, IS]
    # Stored in float32 for accuracy (sigmoid needs higher precision)
    conv_sig = torch.empty((OC, IS), dtype=torch.float32, device=target.device)

    BLOCK_OC = 32
    BLOCK_K = 64
    num_conv_programs = (OC + BLOCK_OC - 1) // BLOCK_OC

    conv1x1_sigmoid_kernel[(num_conv_programs,)](
        weight_ptr=weight,
        input_ptr=input_feat,
        output_ptr=conv_sig,
        OC=OC, IC=IC, IS=IS,
        BLOCK_OC=BLOCK_OC, BLOCK_K=BLOCK_K,
    )

    # Step 2: Bilinear interpolate + multiply -> output [1, OC, OH, OW]
    output = torch.empty_like(target)

    total = OC * OH * OW
    BLOCK_SIZE = 512
    num_interp_programs = (total + BLOCK_SIZE - 1) // BLOCK_SIZE

    # scale_w = W_IN / W_OUT for align_corners=False formula
    scale_w = float(IS) / float(OW)

    interp_mul_kernel[(num_interp_programs,)](
        sig_out_ptr=conv_sig,
        in2_ptr=target,
        output_ptr=output,
        OC=OC, OH=OH, OW=OW, IS=IS,
        scale_w=scale_w,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    return output


def replacement_func():
    return fused_conv_sigmoid_interp_mul