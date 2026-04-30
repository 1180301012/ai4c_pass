import torch
import triton
import triton.language as tl


def pattern(input_tensor, running_mean, running_var, bn_weight, bn_bias, prelu_weight):
    bn_out = torch.nn.functional.batch_norm(input_tensor, running_mean, running_var, bn_weight, bn_bias, False, 0.1, 0.001)
    prelu_out = torch.prelu(bn_out, prelu_weight)
    return prelu_out


def replacement_args(input_tensor, running_mean, running_var, bn_weight, bn_bias, prelu_weight):
    return (input_tensor, running_mean, running_var, bn_weight, bn_bias, prelu_weight)


@triton.jit
def fused_bn_prelu_kernel(
    input_ptr,
    running_mean_ptr,
    running_var_ptr,
    weight_ptr,
    bias_ptr,
    prelu_weight_ptr,
    output_ptr,
    HW,
    C: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # 2D grid: (spatial_blocks, N*C)
    pid_spatial = tl.program_id(0)
    pid_nc = tl.program_id(1)
    c = pid_nc & (C - 1)  # C=128 is power of 2

    # Load channel parameters (once per program) - computed in f32 for precision
    mean_val = tl.load(running_mean_ptr + c).to(tl.float32)
    var_val = tl.load(running_var_ptr + c).to(tl.float32)
    w_val = tl.load(weight_ptr + c).to(tl.float32)
    b_val = tl.load(bias_ptr + c).to(tl.float32)
    alpha_val = tl.load(prelu_weight_ptr + c).to(tl.float32)

    # Precompute BN scale and shift in f32
    inv_std = 1.0 / tl.sqrt(var_val + 0.001)
    scale = inv_std * w_val
    shift = b_val - mean_val * scale

    # Compute offsets for this spatial block
    spatial_start = pid_spatial * BLOCK_SIZE
    offsets = spatial_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < HW

    # Global offset into the tensor
    base_offset = pid_nc * HW
    global_offsets = base_offset + offsets

    # Load input in native dtype, compute in f32 only for the data path
    x = tl.load(input_ptr + global_offsets, mask=mask, other=0.0)
    x_f32 = x.to(tl.float32)

    # Fused BN + PReLU in f32
    bn_out = x_f32 * scale + shift
    out = tl.where(bn_out >= 0.0, bn_out, alpha_val * bn_out)

    # Store (auto-casts back to input dtype)
    tl.store(output_ptr + global_offsets, out, mask=mask)


@torch.fx.wrap
def fused_bn_prelu(input_tensor, running_mean, running_var, bn_weight, bn_bias, prelu_weight):
    N, C, H, W = input_tensor.shape
    HW = H * W

    # Allocate output
    output = torch.empty_like(input_tensor)

    # 2D grid: (spatial_blocks, N*C) - avoids per-element division
    BLOCK_SIZE = 1024
    num_spatial_blocks = (HW + BLOCK_SIZE - 1) // BLOCK_SIZE
    grid = (num_spatial_blocks, N * C)

    fused_bn_prelu_kernel[grid](
        input_tensor,
        running_mean,
        running_var,
        bn_weight,
        bn_bias,
        prelu_weight,
        output,
        HW,
        C,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=4,
    )

    return output


def replacement_func():
    return fused_bn_prelu