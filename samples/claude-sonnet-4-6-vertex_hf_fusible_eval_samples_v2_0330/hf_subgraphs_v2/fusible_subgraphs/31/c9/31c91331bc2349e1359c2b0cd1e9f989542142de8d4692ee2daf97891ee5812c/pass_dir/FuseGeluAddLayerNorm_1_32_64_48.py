import torch
import triton
import triton.language as tl


@triton.jit
def _fused_gelu_add_ln_c32_hw3072(
    in2_ptr, in3_ptr, weight_ptr, bias_ptr,
    out_ptr, ln_out_ptr,
    BLOCK_C: tl.constexpr,
):
    # Each program handles one spatial position (hw index)
    hw = tl.program_id(0)
    HW = 3072  # H*W = 64*48
    C = 32
    EPS = 1e-6

    c_idx = tl.arange(0, BLOCK_C)

    # Load in2: shape [1, C, H, W]; element [0, c, h, w] = ptr[c*HW + hw]
    in2_vals = tl.load(in2_ptr + c_idx * HW + hw).to(tl.float32)

    # Exact GELU: x * 0.5 * (1 + erf(x / sqrt(2)))
    M_SQRT1_2 = 0.7071067811865476
    gelu_vals = in2_vals * 0.5 * (1.0 + tl.math.erf(in2_vals * M_SQRT1_2))

    # Load in3: shape [1, HW, C]; element [0, hw, c] = ptr[hw*C + c]
    in3_vals = tl.load(in3_ptr + hw * BLOCK_C + c_idx).to(tl.float32)

    # Add: gelu(in2) + in3 -> tmp_10
    added = gelu_vals + in3_vals

    # Store tmp_10 (cast back to original dtype)
    tl.store(out_ptr + hw * BLOCK_C + c_idx, added)

    # Layer norm: normalize over C dimension
    mean = tl.sum(added, axis=0) / C
    diff = added - mean
    var = tl.sum(diff * diff, axis=0) / C
    inv_std = tl.math.rsqrt(var + EPS)
    normalized = diff * inv_std

    # Load weight (in_1) and bias (in_0)
    weight = tl.load(weight_ptr + c_idx).to(tl.float32)
    bias_v = tl.load(bias_ptr + c_idx).to(tl.float32)

    ln_out = normalized * weight + bias_v

    # Store layer norm output
    tl.store(ln_out_ptr + hw * BLOCK_C + c_idx, ln_out)


@torch.fx.wrap
def _kernel_launch_c32_hw3072(in_0, in_1, in_2, in_3):
    """Opaque kernel wrapper: returns (tmp_10, tmp_12) tuple."""
    C, H, W = 32, 64, 48
    HW = H * W  # 3072

    tmp_10 = torch.empty(1, HW, C, dtype=in_3.dtype, device=in_3.device)
    ln_out = torch.empty(1, HW, C, dtype=in_3.dtype, device=in_3.device)

    grid = (HW,)
    _fused_gelu_add_ln_c32_hw3072[grid](
        in_2, in_3, in_1, in_0,
        tmp_10, ln_out,
        BLOCK_C=C,
    )

    tmp_12 = ln_out.view(1, H, W, C)
    return (tmp_10, tmp_12)


def _replacement_c32_hw3072(in_0, in_1, in_2, in_3):
    """Non-wrapped so FX traces into it, creating two getitem returning nodes."""
    result = _kernel_launch_c32_hw3072(in_0, in_1, in_2, in_3)
    return result[0], result[1]


def pattern(in_0, in_1, in_2, in_3):
    tmp_2 = torch.nn.functional.gelu(in_2, approximate='none')
    tmp_3 = tmp_2.flatten(2)
    tmp_4 = tmp_3.transpose(1, 2)
    tmp_5 = tmp_4.contiguous()
    tmp_6 = in_3 + tmp_5
    tmp_7 = tmp_6.permute(0, 2, 1)
    tmp_8 = tmp_7.view(1, 32, 64, 48)
    tmp_9 = tmp_8.view(1, 32, -1)
    tmp_10 = tmp_9.permute(0, 2, 1)
    tmp_11 = torch.nn.functional.layer_norm(tmp_10, (32,), in_1, in_0, 1e-06)
    tmp_12 = tmp_11.view(1, 64, 48, 32)
    return (tmp_10, tmp_12)


def replacement_args(in_0, in_1, in_2, in_3):
    return (in_0, in_1, in_2, in_3)


def replacement_func():
    return _replacement_c32_hw3072