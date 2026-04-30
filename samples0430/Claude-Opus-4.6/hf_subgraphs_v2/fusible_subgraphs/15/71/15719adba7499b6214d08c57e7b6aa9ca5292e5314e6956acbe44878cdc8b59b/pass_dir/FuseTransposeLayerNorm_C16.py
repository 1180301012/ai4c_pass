import torch
import triton
import triton.language as tl


def pattern(conv_out, weight, bias):
    tmp_6 = conv_out.flatten(2)
    tmp_7 = tmp_6.transpose(1, 2)
    tmp_8 = torch.nn.functional.layer_norm(tmp_7, (16,), weight, bias, 1e-05)
    tmp_9 = torch.nn.functional.dropout(tmp_8, 0.0, False, False)
    return tmp_9


def replacement_args(conv_out, weight, bias):
    return (conv_out, weight, bias)


@triton.jit
def fused_transpose_layernorm_kernel_16(
    input_ptr,
    weight_ptr,
    bias_ptr,
    output_ptr,
    HW,
    BLOCK_C: tl.constexpr,
):
    # Each program handles one spatial position
    row_idx = tl.program_id(0)
    c_offsets = tl.arange(0, BLOCK_C)
    mask = c_offsets < 16

    # Load from NCHW layout: element at (0, c, h, w) is at offset c*HW + spatial_idx
    x = tl.load(input_ptr + c_offsets * HW + row_idx, mask=mask, other=0.0)
    x_f32 = x.to(tl.float32)

    # Compute mean
    mean = tl.sum(x_f32, axis=0) / 16.0

    # Compute variance
    diff = x_f32 - mean
    var = tl.sum(diff * diff, axis=0) / 16.0

    # Normalize
    inv_std = 1.0 / tl.sqrt(var + 1e-5)
    x_norm = diff * inv_std

    # Apply affine transform (weight and bias)
    w = tl.load(weight_ptr + c_offsets, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(bias_ptr + c_offsets, mask=mask, other=0.0).to(tl.float32)
    out = x_norm * w + b

    # Store in [B, HW, C] layout: position (0, row_idx, c) is at row_idx*C + c
    tl.store(output_ptr + row_idx * 16 + c_offsets, out.to(x.dtype), mask=mask)


@torch.fx.wrap
def fused_transpose_layernorm_16(conv_out, weight, bias):
    B = conv_out.shape[0]
    C = conv_out.shape[1]
    H = conv_out.shape[2]
    W = conv_out.shape[3]
    HW = H * W
    output = torch.empty(B, HW, C, dtype=conv_out.dtype, device=conv_out.device)

    grid = (HW,)
    fused_transpose_layernorm_kernel_16[grid](
        conv_out, weight, bias, output,
        HW=HW, BLOCK_C=16,
    )

    return output


def replacement_func():
    return fused_transpose_layernorm_16