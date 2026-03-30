import torch
import triton
import triton.language as tl


def pattern(key_states, cos, sin, value_states):
    tmp_0 = key_states * cos
    tmp_1 = key_states[Ellipsis, slice(None, 128, None)]
    tmp_2 = key_states[Ellipsis, slice(128, None, None)]
    tmp_3 = -tmp_2
    tmp_4 = torch.cat((tmp_3, tmp_1), dim=-1)
    tmp_5 = tmp_4 * sin
    tmp_6 = tmp_0 + tmp_5
    tmp_10 = value_states[slice(None, None, None), slice(None, None, None), None, slice(None, None, None), slice(None, None, None)]
    tmp_11 = tmp_10.expand(1, 1, 8, 3, 256)
    tmp_12 = tmp_11.reshape(1, 8, 3, 256)
    return tmp_6, tmp_12


def replacement_args(key_states, cos, sin, value_states):
    return (key_states, cos, sin, value_states)


@triton.jit
def rope_pair_kernel(
    key_ptr, cos_ptr, sin_ptr, out_ptr,
    N_pairs,
    D: tl.constexpr,
    HALF_D: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    p = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = p < N_pairs
    seq = p >> 7
    d_lo = p & 127
    flat_lo = seq * D + d_lo
    flat_hi = flat_lo + HALF_D
    k_lo = tl.load(key_ptr + flat_lo, mask=mask, other=0.0)
    k_hi = tl.load(key_ptr + flat_hi, mask=mask, other=0.0)
    c_lo = tl.load(cos_ptr + flat_lo, mask=mask, other=0.0)
    c_hi = tl.load(cos_ptr + flat_hi, mask=mask, other=0.0)
    s_lo = tl.load(sin_ptr + flat_lo, mask=mask, other=0.0)
    s_hi = tl.load(sin_ptr + flat_hi, mask=mask, other=0.0)
    out_lo = k_lo * c_lo + (-k_hi) * s_lo
    out_hi = k_hi * c_hi + k_lo * s_hi
    tl.store(out_ptr + flat_lo, out_lo, mask=mask)
    tl.store(out_ptr + flat_hi, out_hi, mask=mask)


@triton.jit
def expand_v_kernel(
    src_ptr, dst_ptr,
    SD,
    H: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    sd_offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = sd_offs < SD
    vals = tl.load(src_ptr + sd_offs, mask=mask, other=0.0)
    for h in tl.static_range(0, H):
        tl.store(dst_ptr + h * SD + sd_offs, vals, mask=mask)


@torch.fx.wrap
def _rope_kernel_wrapper(key_states, cos, sin):
    B, Hk, S, D = key_states.shape
    HALF_D = D // 2
    N_pairs = B * Hk * S * HALF_D
    k_embed = torch.empty_like(key_states)
    BLOCK_ROPE = 512
    rope_pair_kernel[((N_pairs + BLOCK_ROPE - 1) // BLOCK_ROPE,)](
        key_states, cos, sin, k_embed,
        N_pairs, D, HALF_D, BLOCK_SIZE=BLOCK_ROPE,
    )
    return k_embed


@torch.fx.wrap
def _expand_v_kernel_wrapper(value_states):
    H_out = 8
    B, Hk, S, D = value_states.shape
    SD = S * D
    v_expanded = torch.empty(B, H_out, S, D, dtype=value_states.dtype, device=value_states.device)
    BLOCK_EXP = 256
    expand_v_kernel[((SD + BLOCK_EXP - 1) // BLOCK_EXP,)](
        value_states, v_expanded,
        SD, H_out, BLOCK_SIZE=BLOCK_EXP,
    )
    return v_expanded


def fast_rope_and_vexpand(key_states, cos, sin, value_states):
    k_embed = _rope_kernel_wrapper(key_states, cos, sin)
    v_expanded = _expand_v_kernel_wrapper(value_states)
    return k_embed, v_expanded


def replacement_func():
    return fast_rope_and_vexpand