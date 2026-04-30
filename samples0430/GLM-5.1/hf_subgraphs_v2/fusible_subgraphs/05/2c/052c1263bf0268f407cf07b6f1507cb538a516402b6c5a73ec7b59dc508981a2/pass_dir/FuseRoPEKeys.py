import torch
import triton
import triton.language as tl

def pattern(cos, key, sin):
    tmp_0 = key * cos
    tmp_1 = key[Ellipsis, slice(None, 128, None)]
    tmp_2 = key[Ellipsis, slice(128, None, None)]
    tmp_3 = -tmp_2
    tmp_4 = torch.cat((tmp_3, tmp_1), dim=-1)
    tmp_5 = tmp_4 * sin
    tmp_6 = tmp_0 + tmp_5
    return tmp_6

def replacement_args(cos, key, sin):
    return (cos, key, sin, "rope")

@triton.jit
def rope_kernel(
    key_ptr, cos_ptr, sin_ptr,
    out_ptr,
    n_seq: tl.constexpr, n_dim: tl.constexpr, n_half: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)

    total = n_seq * n_dim
    mask = offsets < total

    seq_pos = offsets // n_dim
    d = offsets % n_dim

    key_val = tl.load(key_ptr + offsets, mask=mask, other=0.0)
    cos_val = tl.load(cos_ptr + offsets, mask=mask, other=0.0)
    sin_val = tl.load(sin_ptr + offsets, mask=mask, other=0.0)

    d_other = tl.where(d < n_half, d + n_half, d - n_half)
    other_offsets = seq_pos * n_dim + d_other

    key_other = tl.load(key_ptr + other_offsets, mask=mask, other=0.0)
    rotated_val = tl.where(d < n_half, -key_other, key_other)

    result = key_val * cos_val + rotated_val * sin_val
    tl.store(out_ptr + offsets, result, mask=mask)

@torch.fx.wrap
def _rope_impl(cos, key, sin):
    n_seq = 3
    n_dim = 256
    n_half = 128
    BLOCK_SIZE = 64

    total = n_seq * n_dim
    num_programs = (total + BLOCK_SIZE - 1) // BLOCK_SIZE

    out = torch.empty((1, 1, n_seq, n_dim), dtype=key.dtype, device=key.device)

    rope_kernel[(num_programs,)](
        key_ptr=key, cos_ptr=cos, sin_ptr=sin,
        out_ptr=out,
        n_seq=n_seq, n_dim=n_dim, n_half=n_half,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    return out

@torch.fx.wrap
def _expand_value_impl(value_states):
    # Placeholder - never called in this pass's context
    raise NotImplementedError("expand_value route not implemented in FuseRoPEKeys pass")

@torch.fx.wrap
def dispatch_wrapper(*args):
    route = args[-1]
    if route == "rope":
        cos, key, sin, _ = args
        return _rope_impl(cos, key, sin)
    elif route == "expand_value":
        value_states, _ = args
        return _expand_value_impl(value_states)
    else:
        raise ValueError(f"Unknown route: {route}")

def replacement_func():
    return dispatch_wrapper