import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Pattern: matches the RoPE computation on in_2 (key_states [1,1,3,256]).
# Produces tmp_6 = rope(key, cos, sin)  shape [1, 1, 3, 256]
# ---------------------------------------------------------------------------
def pattern(in_2, in_1, in_4):
    tmp_0 = in_2 * in_1
    tmp_1 = in_2[Ellipsis, slice(None, 128, None)]
    tmp_2 = in_2[Ellipsis, slice(128, None, None)]
    tmp_3 = -tmp_2
    tmp_4 = torch.cat((tmp_3, tmp_1), dim=-1)
    tmp_5 = tmp_4 * in_4
    tmp_6 = tmp_0 + tmp_5
    return tmp_6


def replacement_args(in_2, in_1, in_4):
    return (in_2, in_1, in_4)


# ---------------------------------------------------------------------------
# Triton kernel: RoPE for all N=3 rows, D=256 each.
# Grid = (3,): one program per sequence position, D=256 elements.
# ---------------------------------------------------------------------------
@triton.jit
def rope_kernel(
    in2_ptr,   # key [1,1,3,256] bf16
    in1_ptr,   # cos [1,1,3,256] bf16
    in4_ptr,   # sin [1,1,3,256] bf16
    out6_ptr,  # rope [1,1,3,256] bf16
    D: tl.constexpr,       # 256
    HALF_D: tl.constexpr,  # 128
):
    n   = tl.program_id(0)   # 0..2
    d   = tl.arange(0, D)    # 0..255

    k1   = tl.load(in2_ptr + n*D + d).to(tl.float32)
    cos1 = tl.load(in1_ptr + n*D + d).to(tl.float32)
    sin1 = tl.load(in4_ptr + n*D + d).to(tl.float32)
    k2   = tl.load(in2_ptr + n*D + HALF_D + d).to(tl.float32)
    cos2 = tl.load(in1_ptr + n*D + HALF_D + d).to(tl.float32)
    sin2 = tl.load(in4_ptr + n*D + HALF_D + d).to(tl.float32)

    rope_hi = k1 * cos1 + k2 * sin1       # d==0..127
    rope_lo = k1 * cos1 + (-k2) * sin1    # d==128..255
    rope    = tl.where(d < HALF_D, rope_lo, rope_hi)

    tl.store(out6_ptr + n*D + d, rope.to(tl.bfloat16))


# ---------------------------------------------------------------------------
# Wrapper — all constants inlined, minimal Python overhead.
# ---------------------------------------------------------------------------
@torch.fx.wrap
def rope_wrapper(in_2, in_1, in_4):
    out6 = torch.empty_like(in_2)
    rope_kernel[(3,)](
        in_2, in_1, in_4,
        out6,
        D=256, HALF_D=128,
        num_warps=4,
    )
    return out6


def replacement_func():
    return rope_wrapper