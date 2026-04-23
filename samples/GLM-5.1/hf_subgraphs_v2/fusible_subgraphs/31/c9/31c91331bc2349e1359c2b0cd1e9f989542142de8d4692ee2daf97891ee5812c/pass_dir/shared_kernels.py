import torch
import triton
import triton.language as tl


@triton.jit
def fused_gelu_add_layernorm_kernel(
    in_2_ptr, in_3_ptr, weight_ptr, bias_ptr, out_sum_ptr, out_ln_ptr,
    n_rows, in_2_stride, eps,
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0)
    if row_idx >= n_rows:
        return

    offsets = tl.arange(0, BLOCK_SIZE)

    # Read in_2 in transposed order: element (c, hw) at offset c * stride + hw
    # in_2 shape is [1, C, H, W] contiguous, so element at (0, c, h, w) = offset c*H*W + h*W + w
    # After flatten(2)+transpose(1,2), we want (hw, c) which maps to c*HW + hw
    in_2_offsets = offsets * in_2_stride + row_idx
    in_2_f32 = tl.load(in_2_ptr + in_2_offsets).to(tl.float32)

    # GELU: x * 0.5 * (1 + erf(x / sqrt(2)))  (approximate='none' = exact)
    sqrt2 = 1.4142135623730951
    gelu_f32 = in_2_f32 * 0.5 * (1.0 + tl.math.erf(in_2_f32 / sqrt2))

    # Read in_3: shape [1, HW, C] contiguous, element (hw, c) at offset hw * C + c
    in_3_offsets = row_idx * BLOCK_SIZE + offsets
    in_3_f32 = tl.load(in_3_ptr + in_3_offsets).to(tl.float32)

    # Sum = gelu + in_3
    sum_f32 = gelu_f32 + in_3_f32

    # Store sum (tmp_10 data) - Triton converts float32 to output dtype
    out_sum_offsets = row_idx * BLOCK_SIZE + offsets
    tl.store(out_sum_ptr + out_sum_offsets, sum_f32)

    # Layer norm with float32 accumulation
    mean = tl.sum(sum_f32, axis=0) / BLOCK_SIZE
    diff = sum_f32 - mean
    var = tl.sum(diff * diff, axis=0) / BLOCK_SIZE
    rstd = 1.0 / tl.sqrt(var + eps)
    norm_f32 = diff * rstd

    weight_f32 = tl.load(weight_ptr + offsets).to(tl.float32)
    bias_f32 = tl.load(bias_ptr + offsets).to(tl.float32)
    out_f32 = norm_f32 * weight_f32 + bias_f32

    # Store LN result - Triton converts float32 to output dtype
    out_ln_offsets = row_idx * BLOCK_SIZE + offsets
    tl.store(out_ln_ptr + out_ln_offsets, out_f32)


@torch.fx.wrap
def fused_gelu_add_layernorm_dispatch(in_2, in_3, weight, bias, route):
    if route == "route_128_16_12":
        C = 128; H = 16; W = 12; BLOCK_SIZE = 128
    elif route == "route_32_64_48":
        C = 32; H = 64; W = 48; BLOCK_SIZE = 32
    elif route == "route_256_8_6":
        C = 256; H = 8; W = 6; BLOCK_SIZE = 256
    else:
        raise ValueError(f"Unknown route: {route}")

    HW = H * W
    in_2_stride = HW

    out_sum = torch.empty((HW, C), dtype=in_2.dtype, device=in_2.device)
    out_ln = torch.empty((HW, C), dtype=in_2.dtype, device=in_2.device)

    fused_gelu_add_layernorm_kernel[(HW,)](
        in_2, in_3, weight, bias, out_sum, out_ln,
        HW, in_2_stride, 1e-06,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    tmp_10 = out_sum.view(1, HW, C)
    tmp_12 = out_ln.view(1, H, W, C)
    return (tmp_10, tmp_12)