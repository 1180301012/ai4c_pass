import torch
import triton
import triton.language as tl


def pattern(in_0, in_1, in_2, in_3, in_4, in_5):
    tmp_4 = in_5 * in_4
    tmp_5 = torch.nn.functional.batch_norm(tmp_4, in_0, in_1, in_3, in_2, False, 0.1, 1e-05)
    tmp_6 = torch.nn.functional.silu(tmp_5, inplace=True)
    return tmp_6


def replacement_args(in_0, in_1, in_2, in_3, in_4, in_5):
    return (in_5, in_4, in_0, in_1, in_3, in_2)


@triton.jit
def fused_mul_bn_silu_kernel(
    x_ptr,
    gate_ptr,
    running_mean_ptr,
    running_var_ptr,
    weight_ptr,
    bias_ptr,
    out_ptr,
    HW,
    C,
    BLOCK_SIZE: tl.constexpr,
):
    pid_hw = tl.program_id(0)
    pid_nc = tl.program_id(1)

    c = pid_nc % C
    n = pid_nc // C

    # Load BN parameters and cast to float32 for precision
    rm = tl.load(running_mean_ptr + c).to(tl.float32)
    rv = tl.load(running_var_ptr + c).to(tl.float32)
    w = tl.load(weight_ptr + c).to(tl.float32)
    b = tl.load(bias_ptr + c).to(tl.float32)

    # Compute BN scale and shift in float32 (inference mode)
    # BN formula: output = weight * (x - mean) / sqrt(var + eps) + bias
    # = x * (weight / sqrt(var + eps)) + (bias - mean * weight / sqrt(var + eps))
    # = x * scale + shift
    scale = w / tl.sqrt(rv + 1e-05)
    shift = b - rm * scale

    # Load gate value and cast to float32
    gate_val = tl.load(gate_ptr + n * C + c).to(tl.float32)

    # Spatial offsets within this block
    spatial_offsets = pid_hw * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = spatial_offsets < HW

    # Flat offsets in NCHW layout
    flat_offsets = pid_nc * HW + spatial_offsets

    # Load input and cast to float32
    x = tl.load(x_ptr + flat_offsets, mask=mask, other=0.0).to(tl.float32)

    # Fused computation in float32: mul * bn * silu
    result = x * gate_val
    result = result * scale + shift
    result = result * tl.sigmoid(result)

    # Store (auto-cast to output dtype)
    tl.store(out_ptr + flat_offsets, result, mask=mask)


@torch.fx.wrap
def fused_mul_bn_silu(x, gate, running_mean, running_var, weight, bias):
    N, C, H, W = x.shape
    HW = H * W
    NC = N * C

    # Move BN parameters to same device as input using allowed API
    device = x.device
    running_mean = torch.as_tensor(running_mean, device=device)
    running_var = torch.as_tensor(running_var, device=device)
    weight = torch.as_tensor(weight, device=device)
    bias = torch.as_tensor(bias, device=device)

    out = torch.empty_like(x)

    BLOCK_SIZE = 1024
    grid = ((HW + BLOCK_SIZE - 1) // BLOCK_SIZE, NC)

    fused_mul_bn_silu_kernel[grid](
        x_ptr=x, gate_ptr=gate,
        running_mean_ptr=running_mean, running_var_ptr=running_var,
        weight_ptr=weight, bias_ptr=bias,
        out_ptr=out,
        HW=HW, C=C,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    return out


def replacement_func():
    return fused_mul_bn_silu