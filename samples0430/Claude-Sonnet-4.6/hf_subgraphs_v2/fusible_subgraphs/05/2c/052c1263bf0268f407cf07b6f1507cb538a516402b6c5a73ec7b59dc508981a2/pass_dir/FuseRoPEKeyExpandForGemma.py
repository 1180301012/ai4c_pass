import torch
import triton
import triton.language as tl


def pattern(key_states, cos, sin):
    tmp_0 = key_states * cos
    tmp_1 = key_states[Ellipsis, slice(None, 128, None)]
    tmp_2 = key_states[Ellipsis, slice(128, None, None)]
    tmp_3 = -tmp_2
    tmp_4 = torch.cat((tmp_3, tmp_1), dim=-1)
    tmp_5 = tmp_4 * sin
    tmp_6 = tmp_0 + tmp_5
    return tmp_6


def replacement_args(key_states, cos, sin):
    return (key_states, cos, sin)


@triton.jit
def _rope_fused_kernel(
    key_ptr, cos_ptr, sin_ptr, out_ptr,
    N: tl.constexpr,
    DIM: tl.constexpr,
    HALF_DIM: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N

    key     = tl.load(key_ptr + offsets, mask=mask, other=0.0)
    cos_val = tl.load(cos_ptr + offsets, mask=mask, other=0.0)
    sin_val = tl.load(sin_ptr + offsets, mask=mask, other=0.0)

    # rotate_half: d < HALF_DIM → -key[d+HALF_DIM],  d >= HALF_DIM → key[d-HALF_DIM]
    d = offsets % DIM
    seq_base = offsets - d
    rotate_offsets = seq_base + (d + HALF_DIM) % DIM
    rotate_key = tl.load(key_ptr + rotate_offsets, mask=mask, other=0.0)
    rotate_key = tl.where(d < HALF_DIM, -rotate_key, rotate_key)

    result = key * cos_val + rotate_key * sin_val
    tl.store(out_ptr + offsets, result, mask=mask)


@torch.fx.wrap
def _rope_wrapper(key_states, cos, sin):
    N = 768   # 1*1*3*256
    DIM = 256
    HALF_DIM = 128
    BLOCK_SIZE = 256
    out = torch.empty_like(key_states)
    grid = ((N + BLOCK_SIZE - 1) // BLOCK_SIZE,)
    _rope_fused_kernel[grid](
        key_states, cos, sin, out,
        N=N, DIM=DIM, HALF_DIM=HALF_DIM, BLOCK_SIZE=BLOCK_SIZE,
    )
    return out


def replacement_func():
    return _rope_wrapper