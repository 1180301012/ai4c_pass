import torch
import triton
import triton.language as tl


@triton.jit
def fused_mul_add_kernel(
    in0_ptr,   # [2, 128]        strides (128, 1)
    in1_ptr,   # [1, 1, 2, 128]  flat: k2*128 + j
    in2_ptr,   # [B, 17, 1, 128] flat: b*K2J + k*J + j  (K2J=2176, J=128)
    out_ptr,   # [B, 17, 2, 128] flat: b*K2J*2 + k*K2*J + k2*J + j  (K2=2)
    BLOCK_J: tl.constexpr,  # = 128
):
    """
    Flat 1D grid: one program per BLOCK_J=128 consecutive flat output positions.
    Each flat position encodes (b, k, k2, j).
    in0/in1 loads: k = k2//2 (pure scalar, no vector division).
    in2/out stores: explicitly computed in-bounds. No vector integer divisions.
    """
    pid  = tl.program_id(0)
    j    = tl.arange(0, BLOCK_J)            # j = 0..127

    # Scalar decode: pid // 34 = b, pid % 34 = kt (k = kt//2, k2 = kt%2)
    b    = pid // 34            # scalar batch index
    kt   = pid %  34            # 0..33
    k    = kt // 2              # K dimension  (0..16)
    k2   = kt %  2              # K2 dimension (0 or 1)

    # Loads
    scale = tl.load(in1_ptr + k2 * 128 + j) # [2, 128]  scalar k2, stride-1 load ✓
    bias  = tl.load(in0_ptr + k2 * 128 + j) # [2, 128]
    val   = tl.load(in2_ptr + b * 2176 + k * 128 + j) # [128] ✓ max b*2176 + 17*128 + 127 = B*KJ - 1

    result = val * scale + bias

    # Write both k2=0 and k2=1 output rows
    out_off = b * 4352 + k * 256 + k2 * 128   # = b*K2J + k*K2*J + k2*J
    tl.store(out_ptr + out_off            + j, result)
    tl.store(out_ptr + out_off + 128      + j, result)


@torch.fx.wrap
def fused_mul_add(in_0, in_1, in_2):
    """
    tmp_2 = in_2 * in_1 + in_0  →  [B, 17, 2, 128].
    Returns [B, 17, 2, 128] so torch.unbind(dim=2) gives 2 tensors of shape [B, 128, 17].
    Static (pre-computed) grid avoids Python overhead per call.
    """
    B   = in_2.shape[0]
    out = torch.empty((B, 17, 2, 128), dtype=in_2.dtype, device=in_2.device)

    BLOCK_J = 128
    grid    = (B * 34,)   # B * K * K2 = B * 17 * 2 = B * 34  (static tuple)

    fused_mul_add_kernel[grid](
        in_0, in_1, in_2, out,
        BLOCK_J=BLOCK_J,
        num_warps=4,
    )

    return out


# ── Pattern ──────────────────────────────────────────────────────────────────

def pattern(in_0, in_1, in_2):
    """Match: tmp_2 = in_2 * in_1 + in_0."""
    tmp_1 = in_2 * in_1
    tmp_2 = tmp_1 + in_0
    return tmp_2


def replacement_args(in_0, in_1, in_2):
    return (in_0, in_1, in_2)


def replacement_func():
    return fused_mul_add