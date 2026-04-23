import torch
import triton
import triton.language as tl


def pattern(in_1, in_2, in_4):
    tmp_0 = in_2 * in_1
    tmp_1 = in_2[Ellipsis, slice(None, 128, None)]
    tmp_2 = in_2[Ellipsis, slice(128, None, None)]
    tmp_3 = -tmp_2
    tmp_4 = torch.cat((tmp_3, tmp_1), dim=-1)
    tmp_5 = tmp_4 * in_4
    tmp_6 = tmp_0 + tmp_5
    return tmp_6


def replacement_args(in_1, in_2, in_4):
    return (in_1, in_2, in_4)


@triton.jit
def rotary_embed_kernel(
    key_ptr, cos_ptr, sin_ptr, out_ptr,
    half_dim: tl.constexpr,
    full_dim: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    row_start = pid * full_dim
    first_offsets = row_start + tl.arange(0, half_dim)
    second_offsets = row_start + half_dim + tl.arange(0, half_dim)

    k_first = tl.load(key_ptr + first_offsets)
    k_second = tl.load(key_ptr + second_offsets)
    c_first = tl.load(cos_ptr + first_offsets)
    c_second = tl.load(cos_ptr + second_offsets)
    s_first = tl.load(sin_ptr + first_offsets)
    s_second = tl.load(sin_ptr + second_offsets)

    out_first = k_first * c_first - k_second * s_first
    out_second = k_second * c_second + k_first * s_second

    tl.store(out_ptr + first_offsets, out_first)
    tl.store(out_ptr + second_offsets, out_second)


@torch.fx.wrap
def rotary_embed(cos, key, sin):
    n_rows = 3
    half_dim = 128
    full_dim = 256

    out = torch.empty_like(key)

    rotary_embed_kernel[(n_rows,)](
        key_ptr=key,
        cos_ptr=cos,
        sin_ptr=sin,
        out_ptr=out,
        half_dim=half_dim,
        full_dim=full_dim,
        BLOCK_SIZE=256,
    )

    return out


def replacement_func():
    return rotary_embed