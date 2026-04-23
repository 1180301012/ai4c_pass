import torch
import triton
import triton.language as tl
import math

def pattern(in_0, in_1, in_2, in_3, in_4, in_5):
    tmp_4 = in_5 * in_4
    tmp_5 = torch.nn.functional.batch_norm(tmp_4, in_0, in_1, in_3, in_2, False, 0.1, 1e-05)
    tmp_6 = torch.nn.functional.silu(tmp_5, inplace=True)
    return (tmp_6,)

def replacement_args(in_0, in_1, in_2, in_3, in_4, in_5):
    return (in_0, in_1, in_2, in_3, in_4, in_5)

@triton.jit
def fused_mul_bn_silu_kernel(
    x_ptr,          # in_5: main input [N, C, H, W]
    gate_ptr,       # in_4: sigmoid gate [N, C, 1, 1] or broadcastable
    bn_mean_ptr,    # in_0: BN running mean [C]
    bn_var_ptr,     # in_1: BN running var [C]
    bn_weight_ptr,  # in_3: BN weight (gamma) [C]
    bn_bias_ptr,    # in_2: BN bias (beta) [C]
    out_ptr,        # output [N, C, H, W]
    N: tl.constexpr,
    C: tl.constexpr,
    HW: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_HW: tl.constexpr,
):
    # BN in inference: out = gamma * (x - mean) / sqrt(var + eps) + beta
    # which simplifies to: out = scale_c * x + offset_c
    # where scale_c = gamma / sqrt(var + eps), offset_c = beta - gamma * mean / sqrt(var + eps)
    eps = 1e-5

    pid_n = tl.program_id(0)
    pid_c = tl.program_id(1)

    # Compute BN scale and offset for this channel
    mean_val = tl.load(bn_mean_ptr + pid_c)
    var_val = tl.load(bn_var_ptr + pid_c)
    weight_val = tl.load(bn_weight_ptr + pid_c)
    bias_val = tl.load(bn_bias_ptr + pid_c)

    inv_std = 1.0 / tl.sqrt(var_val + eps)
    bn_scale = weight_val * inv_std
    bn_offset = bias_val - weight_val * mean_val * inv_std

    # Load gate value for this (n, c)
    gate_val = tl.load(gate_ptr + pid_n * C + pid_c)

    # Process spatial dimensions in blocks
    for hw_start in range(0, HW, BLOCK_HW):
        hw_offsets = hw_start + tl.arange(0, BLOCK_HW)
        hw_mask = hw_offsets < HW

        x_offsets = pid_n * C * HW + pid_c * HW + hw_offsets

        # Load input
        x_val = tl.load(x_ptr + x_offsets, mask=hw_mask, other=0.0)

        # Step 1: multiply with gate
        gated = x_val * gate_val

        # Step 2: apply BN (inference mode)
        normed = bn_scale * gated + bn_offset

        # Step 3: apply SiLU: x * sigmoid(x)
        # sigmoid(x) = 1 / (1 + exp(-x))
        # For numerical stability, handle large negative values
        silu_out = normed * tl.sigmoid(normed)

        # Store output
        tl.store(out_ptr + x_offsets, silu_out, mask=hw_mask)


@torch.fx.wrap
def fused_mul_bn_silu(x, gate, bn_mean, bn_var, bn_bias, bn_weight):
    # x: [N, C, H, W], gate: [N, C, 1, 1] (broadcastable), BN params: [C]
    N_dim = x.shape[0]
    C_dim = x.shape[1]
    H_dim = x.shape[2]
    W_dim = x.shape[3]
    HW = H_dim * W_dim

    out = torch.empty_like(x)

    # Choose block size for spatial dimension
    BLOCK_HW = min(1024, triton.next_power_of_2(HW))

    grid = (N_dim, C_dim)

    fused_mul_bn_silu_kernel[grid](
        x_ptr=x,
        gate_ptr=gate.reshape(N_dim, C_dim),  # squeeze spatial dims for gate
        bn_mean_ptr=bn_mean,
        bn_var_ptr=bn_var,
        bn_weight_ptr=bn_weight,
        bn_bias_ptr=bn_bias,
        out_ptr=out,
        N=N_dim,
        C=C_dim,
        HW=HW,
        BLOCK_N=1,
        BLOCK_HW=BLOCK_HW,
    )

    return out

def replacement_func():
    return fused_mul_bn_silu