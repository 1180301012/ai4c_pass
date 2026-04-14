import operator
import torch
import triton
import triton.language as tl


def pattern(in_5, in_1, in_2, in_4, in_3):
    tmp_6 = torch.nn.functional.adaptive_avg_pool2d(in_5, (1, 1))
    tmp_7 = torch.nn.functional.batch_norm(tmp_6, in_1, in_2, in_4, in_3, False, 0.1, 1e-05)
    return tmp_7


def replacement_args(in_5, in_1, in_2, in_4, in_3):
    return (in_5, in_1, in_2, in_4, in_3)


@triton.jit
def fused_avgpool_bn_relu_kernel(
    input_ptr,
    mean_ptr,
    var_ptr,
    weight_ptr,
    bias_ptr,
    output_ptr,
    B, C, C_BLOCKS, HW: tl.constexpr,
    IS_FP16: tl.constexpr,
    IS_BF16: tl.constexpr,
):
    # Grid: B * C_BLOCKS programs; each handles 64 channels for one batch item
    pid    = tl.program_id(0)
    b      = pid // C_BLOCKS
    cb     = pid % C_BLOCKS
    c_start = cb * 64

    c_offsets  = c_start + tl.arange(0, 64)
    hw_offsets = tl.arange(0, HW)

    # BN parameters for 64 channels
    running_mean = tl.load(mean_ptr   + c_offsets).to(tl.float32)
    running_var  = tl.load(var_ptr    + c_offsets).to(tl.float32)
    bn_weight    = tl.load(weight_ptr + c_offsets).to(tl.float32)
    bn_bias      = tl.load(bias_ptr   + c_offsets).to(tl.float32)

    # Load [64, HW] input block, compute per-channel mean (avg pool)
    base = b * C * HW + c_start * HW
    ptrs = input_ptr + base + tl.arange(0, 64)[:, None] * HW + hw_offsets[None, :]
    vals = tl.load(ptrs).to(tl.float32)
    avg  = tl.sum(vals, axis=1) * (1.0 / HW)  # [64]

    # BN inference
    inv_std = 1.0 / tl.sqrt(running_var + 1e-5)
    out = (avg - running_mean) * inv_std * bn_weight + bn_bias

    # Store [64] results to output [B, C, 1, 1]
    out_ptrs = output_ptr + b * C + c_offsets
    if IS_FP16:
        tl.store(out_ptrs, out.to(tl.float16))
    elif IS_BF16:
        tl.store(out_ptrs, out.to(tl.bfloat16))
    else:
        tl.store(out_ptrs, out.to(tl.float32))


@torch.fx.wrap
def fused_avgpool_bn_relu(in_5, in_1, in_2, in_4, in_3):
    B, C, H, W = in_5.shape
    HW       = H * W
    C_BLOCKS = C // 64       # 64 channels per block; C=512 → 8 blocks per batch
    BC       = B * C
    output   = torch.empty((B, C, 1, 1), dtype=in_5.dtype, device=in_5.device)

    is_fp16 = (in_5.dtype == torch.float16)
    is_bf16 = (in_5.dtype == torch.bfloat16)

    fused_avgpool_bn_relu_kernel[(B * C_BLOCKS,)](
        in_5, in_1, in_2, in_4, in_3, output,
        B, C, C_BLOCKS, HW,
        IS_FP16=is_fp16,
        IS_BF16=is_bf16,
        num_warps=4,
    )
    return output


def replacement_func():
    return fused_avgpool_bn_relu