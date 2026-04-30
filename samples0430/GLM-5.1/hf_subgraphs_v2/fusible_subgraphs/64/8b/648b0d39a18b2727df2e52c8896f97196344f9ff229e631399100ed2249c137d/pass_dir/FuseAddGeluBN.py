import torch
import triton
import triton.language as tl


def pattern(in_0, in_1, in_2, in_3, in_4, in_5):
    in_4 += in_5
    in_6 = in_4
    tmp_5 = torch.nn.functional.gelu(in_6, approximate='none')
    tmp_6 = torch.nn.functional.batch_norm(tmp_5, in_0, in_1, in_3, in_2, False, 0.1, 1e-05)
    tmp_7 = 0 + tmp_6
    return (tmp_5, tmp_7)


def replacement_args(in_0, in_1, in_2, in_3, in_4, in_5):
    # in_0 = running_mean, in_1 = running_var, in_2 = bias, in_3 = weight
    # kernel takes: in_4, in_5, running_mean, running_var, weight, bias
    return (in_4, in_5, in_0, in_1, in_3, in_2)


@triton.jit
def fused_add_gelu_bn_kernel(
    in_4_ptr, in_5_ptr,
    running_mean_ptr, running_var_ptr, weight_ptr, bias_ptr,
    gelu_out_ptr, bn_out_ptr,
    n_elements,
    C,
    HW,
    eps: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # Load input elements and cast to float32 for numerical stability
    in_4_val = tl.load(in_4_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    in_5_val = tl.load(in_5_ptr + offsets, mask=mask, other=0.0).to(tl.float32)

    # Step 1: Add
    x = in_4_val + in_5_val

    # Step 2: GELU (exact: gelu(x) = x * 0.5 * (1 + erf(x / sqrt(2))))
    sqrt2 = 1.4142135623730951
    gelu_result = x * 0.5 * (1.0 + tl.math.erf(x / sqrt2))

    # Determine channel index for each element
    # For contiguous 4D tensor [B, C, H, W]: channel_idx = (flat_index // HW) % C
    channel_idx = (offsets // HW) % C

    # Load BN parameters and cast to float32 for numerical stability
    running_mean = tl.load(running_mean_ptr + channel_idx, mask=mask, other=0.0).to(tl.float32)
    running_var = tl.load(running_var_ptr + channel_idx, mask=mask, other=1.0).to(tl.float32)
    weight_val = tl.load(weight_ptr + channel_idx, mask=mask, other=1.0).to(tl.float32)
    bias_val = tl.load(bias_ptr + channel_idx, mask=mask, other=0.0).to(tl.float32)

    # Step 3: Batch Norm
    # bn(x) = (x - mean) / sqrt(var + eps) * weight + bias
    # Optimized: bn(x) = x * (weight / sqrt(var + eps)) + (bias - mean * weight / sqrt(var + eps))
    inv_std = 1.0 / tl.math.sqrt(running_var + eps)
    scale = weight_val * inv_std
    bn_offset = bias_val - running_mean * scale
    bn_result = gelu_result * scale + bn_offset

    # Step 4: 0 + bn_result = bn_result (identity operation)

    # Store outputs (Triton handles dtype conversion from float32 to output dtype)
    tl.store(gelu_out_ptr + offsets, gelu_result, mask=mask)
    tl.store(bn_out_ptr + offsets, bn_result, mask=mask)


@torch.fx.wrap
def fused_add_gelu_bn(in_4, in_5, running_mean, running_var, weight, bias):
    B, C, H, W = in_4.shape
    HW = H * W
    n_elements = in_4.numel()

    gelu_out = torch.empty_like(in_4)
    bn_out = torch.empty_like(in_4)

    BLOCK_SIZE = 1024
    num_programs = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE

    fused_add_gelu_bn_kernel[(num_programs,)](
        in_4, in_5,
        running_mean, running_var, weight, bias,
        gelu_out, bn_out,
        n_elements=n_elements,
        C=C, HW=HW,
        eps=1e-05,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    return gelu_out, bn_out


def replacement_func():
    return fused_add_gelu_bn