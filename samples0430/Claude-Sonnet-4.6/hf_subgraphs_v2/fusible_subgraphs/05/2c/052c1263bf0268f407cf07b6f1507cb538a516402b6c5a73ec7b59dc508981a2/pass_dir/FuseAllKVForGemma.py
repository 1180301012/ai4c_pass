import torch
import triton
import triton.language as tl


def pattern(key_states, cos, sin, value_states):
    # RoPE for key_states
    tmp_0 = key_states * cos
    tmp_1 = key_states[Ellipsis, slice(None, 128, None)]
    tmp_2 = key_states[Ellipsis, slice(128, None, None)]
    tmp_3 = -tmp_2
    tmp_4 = torch.cat((tmp_3, tmp_1), dim=-1)
    tmp_5 = tmp_4 * sin
    tmp_6 = tmp_0 + tmp_5
    # expand k_embed -> k_expand [1,8,3,256]
    tmp_7 = tmp_6[slice(None, None, None), slice(None, None, None), None, slice(None, None, None), slice(None, None, None)]
    tmp_8 = tmp_7.expand(1, 1, 8, 3, 256)
    tmp_9 = tmp_8.reshape(1, 8, 3, 256)
    # expand value_states -> v_expand [1,8,3,256]
    tmp_10 = value_states[slice(None, None, None), slice(None, None, None), None, slice(None, None, None), slice(None, None, None)]
    tmp_11 = tmp_10.expand(1, 1, 8, 3, 256)
    tmp_12 = tmp_11.reshape(1, 8, 3, 256)
    return tmp_6, tmp_9, tmp_12


def replacement_args(key_states, cos, sin, value_states):
    return (key_states, cos, sin, value_states)


@triton.jit
def _mega_kv_kernel(
    key_ptr, cos_ptr, sin_ptr, value_ptr,
    k_embed_ptr, k_expand_ptr, v_expand_ptr,
    N: tl.constexpr,           # SEQ_LEN * DIM = 768
    DIM: tl.constexpr,         # 256
    HALF_DIM: tl.constexpr,    # 128
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N

    # Load inputs (bfloat16)
    key     = tl.load(key_ptr   + offsets, mask=mask, other=0.0)
    cos_val = tl.load(cos_ptr   + offsets, mask=mask, other=0.0)
    sin_val = tl.load(sin_ptr   + offsets, mask=mask, other=0.0)
    val     = tl.load(value_ptr + offsets, mask=mask, other=0.0)

    # rotate_half: for d < HALF_DIM use -key[seq*DIM + d+HALF_DIM]
    #              for d >= HALF_DIM use key[seq*DIM + d-HALF_DIM]
    d = offsets % DIM
    seq_base = offsets - d
    rotate_offsets = seq_base + (d + HALF_DIM) % DIM
    rotate_key = tl.load(key_ptr + rotate_offsets, mask=mask, other=0.0)
    rotate_key = tl.where(d < HALF_DIM, -rotate_key, rotate_key)

    # k_embed = key * cos + rotate_half(key) * sin
    k_embed = key * cos_val + rotate_key * sin_val

    # --- Write k_embed [1,1,3,256] ---
    tl.store(k_embed_ptr + offsets, k_embed, mask=mask)

    # --- Broadcast k_embed -> k_expand [1,8,3,256]: head h at offset h*N ---
    tl.store(k_expand_ptr + 0 * N + offsets, k_embed, mask=mask)
    tl.store(k_expand_ptr + 1 * N + offsets, k_embed, mask=mask)
    tl.store(k_expand_ptr + 2 * N + offsets, k_embed, mask=mask)
    tl.store(k_expand_ptr + 3 * N + offsets, k_embed, mask=mask)
    tl.store(k_expand_ptr + 4 * N + offsets, k_embed, mask=mask)
    tl.store(k_expand_ptr + 5 * N + offsets, k_embed, mask=mask)
    tl.store(k_expand_ptr + 6 * N + offsets, k_embed, mask=mask)
    tl.store(k_expand_ptr + 7 * N + offsets, k_embed, mask=mask)

    # --- Broadcast value -> v_expand [1,8,3,256] ---
    tl.store(v_expand_ptr + 0 * N + offsets, val, mask=mask)
    tl.store(v_expand_ptr + 1 * N + offsets, val, mask=mask)
    tl.store(v_expand_ptr + 2 * N + offsets, val, mask=mask)
    tl.store(v_expand_ptr + 3 * N + offsets, val, mask=mask)
    tl.store(v_expand_ptr + 4 * N + offsets, val, mask=mask)
    tl.store(v_expand_ptr + 5 * N + offsets, val, mask=mask)
    tl.store(v_expand_ptr + 6 * N + offsets, val, mask=mask)
    tl.store(v_expand_ptr + 7 * N + offsets, val, mask=mask)


@torch.fx.wrap
def _mega_kv_wrapper(key_states, cos, sin, value_states):
    SEQ_LEN = 3
    DIM = 256
    HALF_DIM = 128
    N = SEQ_LEN * DIM  # 768

    k_embed  = torch.empty_like(key_states)                               # [1,1,3,256]
    k_expand = torch.empty((1, 8, SEQ_LEN, DIM), dtype=key_states.dtype,
                           device=key_states.device)                       # [1,8,3,256]
    v_expand = torch.empty((1, 8, SEQ_LEN, DIM), dtype=value_states.dtype,
                           device=value_states.device)                     # [1,8,3,256]

    BLOCK_SIZE = 256
    grid = ((N + BLOCK_SIZE - 1) // BLOCK_SIZE,)   # 3 programs

    _mega_kv_kernel[grid](
        key_states, cos, sin, value_states,
        k_embed, k_expand, v_expand,
        N=N, DIM=DIM, HALF_DIM=HALF_DIM,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    return k_embed, k_expand, v_expand


def replacement_func():
    return _mega_kv_wrapper