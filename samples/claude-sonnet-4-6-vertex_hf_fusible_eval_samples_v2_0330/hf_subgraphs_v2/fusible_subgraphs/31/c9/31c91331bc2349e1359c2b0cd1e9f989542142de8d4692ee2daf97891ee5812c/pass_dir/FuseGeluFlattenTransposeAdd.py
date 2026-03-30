import torch
import triton
import triton.language as tl


@triton.jit
def _fused_gelu_add_kernel(
    in2_ptr,   # [1, C, H, W]  channel-major
    in3_ptr,   # [1, HW, C]    spatial-major
    out_ptr,   # [1, HW, C]    spatial-major output
    C,         # number of channels (runtime)
    HW,        # H*W spatial size   (runtime)
    BLOCK_C: tl.constexpr,
):
    """
    Fuses: GELU(in2) flattened+transposed (channel->spatial) + in3.
    Each program handles one spatial position (hw index).
    in2[0, c, hw] is at ptr offset c*HW + hw  (channel-major layout).
    in3[0, hw, c] is at ptr offset hw*C + c   (spatial-major layout).
    """
    hw = tl.program_id(0)

    c_idx = tl.arange(0, BLOCK_C)
    mask = c_idx < C

    # Load C values from in2 (strided: channel-major, stride HW between channels)
    in2_vals = tl.load(in2_ptr + c_idx * HW + hw, mask=mask, other=0.0).to(tl.float32)

    # Exact GELU: x * 0.5 * (1 + erf(x / sqrt(2)))
    M_SQRT1_2 = 0.7071067811865476
    gelu_vals = in2_vals * 0.5 * (1.0 + tl.math.erf(in2_vals * M_SQRT1_2))

    # Load C values from in3 (contiguous: spatial-major)
    in3_vals = tl.load(in3_ptr + hw * C + c_idx, mask=mask, other=0.0).to(tl.float32)

    # Add
    result = gelu_vals + in3_vals

    # Store spatial-major output: out[0, hw, c] = result
    tl.store(out_ptr + hw * C + c_idx, result, mask=mask)


@torch.fx.wrap
def _fused_gelu_add(in_2, in_3):
    """
    Fused GELU + channel-to-spatial transpose + residual add.
    in_2: [1, C, H, W]  — activation (channel-major)
    in_3: [1, H*W, C]   — residual (spatial-major)
    Returns: [1, H*W, C] contiguous tensor
    """
    B, C, H, W = in_2.shape
    HW = H * W

    out = torch.empty_like(in_3)

    # BLOCK_C must be a power of 2 >= C for tl.arange correctness
    BLOCK_C = max(32, triton.next_power_of_2(C))
    # One thread per channel element → num_warps = BLOCK_C / 32
    num_warps = max(1, BLOCK_C // 32)

    grid = (HW,)
    _fused_gelu_add_kernel[grid](
        in_2, in_3, out,
        C, HW,
        BLOCK_C=BLOCK_C,
        num_warps=num_warps,
    )
    return out


def pattern(in_2, in_3):
    """
    Shape-agnostic pattern: matches GELU + flatten + transpose + contiguous + add
    across ALL graph variants (C=32/128/256), regardless of H, W dimensions.
    """
    tmp_2 = torch.nn.functional.gelu(in_2, approximate='none')
    tmp_3 = tmp_2.flatten(2)
    tmp_4 = tmp_3.transpose(1, 2)
    tmp_5 = tmp_4.contiguous()
    tmp_6 = in_3 + tmp_5
    return tmp_6


def replacement_args(in_2, in_3):
    return (in_2, in_3)


def replacement_func():
    return _fused_gelu_add