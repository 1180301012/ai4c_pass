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


@triton.jit
def fused_conv_sigmoid_interp_mul_kernel(
    weight_ptr, input_ptr, in2_ptr, out_ptr,
    IS_FP16: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fused kernel: conv1x1 + sigmoid + bilinear_interpolate + multiply
    Grid: (128, num_spatial_blocks)
    
    weight: [128, 960, 1, 1] stored as [128, 960] contiguous
    input: [1, 960, 1, 4] stored as [960, 4] contiguous  
    in2: [1, 128, 64, 128] stored as [128, 8192] contiguous
    output: [1, 128, 64, 128] stored as [128, 8192] contiguous
    """
    pid_c = tl.program_id(0)  # output channel [0, 127]
    pid_s = tl.program_id(1)  # spatial block index

    # ===== Phase 1: Conv1x1 + Sigmoid =====
    # Compute 4 dot products: out[c, w] = sum_k weight[c, k] * input[k, w]
    K: tl.constexpr = 960
    acc0 = 0.0
    acc1 = 0.0
    acc2 = 0.0
    acc3 = 0.0

    for k_start in range(0, K, BLOCK_K):
        k_offs = k_start + tl.arange(0, BLOCK_K)
        k_mask = k_offs < K

        # Load weight[pid_c, k_offs]
        w = tl.load(weight_ptr + pid_c * K + k_offs, mask=k_mask, other=0.0).to(tl.float32)

        # Load input[k_offs, 0..3]
        inp0 = tl.load(input_ptr + k_offs * 4, mask=k_mask, other=0.0).to(tl.float32)
        inp1 = tl.load(input_ptr + k_offs * 4 + 1, mask=k_mask, other=0.0).to(tl.float32)
        inp2 = tl.load(input_ptr + k_offs * 4 + 2, mask=k_mask, other=0.0).to(tl.float32)
        inp3 = tl.load(input_ptr + k_offs * 4 + 3, mask=k_mask, other=0.0).to(tl.float32)

        acc0 += tl.sum(w * inp0)
        acc1 += tl.sum(w * inp1)
        acc2 += tl.sum(w * inp2)
        acc3 += tl.sum(w * inp3)

    # Apply sigmoid
    s0 = tl.sigmoid(acc0)
    s1 = tl.sigmoid(acc1)
    s2 = tl.sigmoid(acc2)
    s3 = tl.sigmoid(acc3)

    # ===== Phase 2: Bilinear Interpolation + Multiply =====
    # Interpolate from [1, 4] to [64, 128] in width (height=1 is trivial)
    # scale_w = 4/128 = 0.03125

    spatial_start = pid_s * BLOCK_SIZE
    spatial_offs = spatial_start + tl.arange(0, BLOCK_SIZE)
    mask = spatial_offs < 8192  # H_out * W_out = 64 * 128

    # Output width position
    w_out = spatial_offs % 128

    # Source width coordinate (bilinear, align_corners=False)
    src_w = (w_out.to(tl.float32) + 0.5) * 0.03125 - 0.5

    w0 = tl.floor(src_w).to(tl.int32)
    w1 = w0 + 1
    lerp_w = src_w - w0.to(tl.float32)

    # Clamp indices to [0, 3]
    w0_c = tl.minimum(tl.maximum(w0, 0), 3)
    w1_c = tl.minimum(tl.maximum(w1, 0), 3)

    # Gather sigmoid values using conditional selection
    val_w0 = tl.where(w0_c == 0, s0,
             tl.where(w0_c == 1, s1,
             tl.where(w0_c == 2, s2, s3)))

    val_w1 = tl.where(w1_c == 0, s0,
             tl.where(w1_c == 1, s1,
             tl.where(w1_c == 2, s2, s3)))

    # Bilinear interpolation (1D in width only)
    interp_val = val_w0 * (1.0 - lerp_w) + val_w1 * lerp_w

    # Load in_2 and multiply
    base_offset = pid_c * 8192
    in2_val = tl.load(in2_ptr + base_offset + spatial_offs, mask=mask, other=0.0).to(tl.float32)
    result = in2_val * interp_val

    # Store result with appropriate dtype
    if IS_FP16:
        tl.store(out_ptr + base_offset + spatial_offs, result.to(tl.float16), mask=mask)
    else:
        tl.store(out_ptr + base_offset + spatial_offs, result.to(tl.bfloat16), mask=mask)


@torch.fx.wrap
def fused_conv_sigmoid_interp_mul(in_0, in_1, in_2):
    """
    in_0: weight [128, 960, 1, 1]
    in_1: input [1, 960, 1, 4]
    in_2: feature map [1, 128, 64, 128]
    output: [1, 128, 64, 128]
    """
    output = torch.empty_like(in_2)
    is_fp16 = (in_2.dtype == torch.float16)

    BLOCK_SIZE = 1024
    num_spatial_blocks = 8  # 8192 / 1024

    fused_conv_sigmoid_interp_mul_kernel[(128, num_spatial_blocks)](
        in_0, in_1, in_2, output,
        IS_FP16=is_fp16,
        BLOCK_K=256,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    return output


def replacement_func():
    return fused_conv_sigmoid_interp_mul