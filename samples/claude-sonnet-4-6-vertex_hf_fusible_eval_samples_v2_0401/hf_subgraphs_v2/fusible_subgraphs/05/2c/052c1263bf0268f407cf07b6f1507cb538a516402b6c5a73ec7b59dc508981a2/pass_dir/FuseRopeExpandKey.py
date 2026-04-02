import torch
import triton
import triton.language as tl


@triton.jit
def rope_expand_val_kernel(
    key_ptr,
    cos_ptr,
    sin_ptr,
    val_ptr,
    out_rope_ptr,
    out_expand_ptr,
    out_val_ptr,
    total_seq,
    HEAD_DIM: tl.constexpr,    # 256
    N_REP:    tl.constexpr,    # 8
    BLOCK_SIZE: tl.constexpr,  # 256 (= HEAD_DIM)
):
    """
    Fully fused kernel: RoPE on key + expand key + expand value states.

    Grid: (total_seq,) = (3,) — one CTA per sequence position.
    Each CTA:
      1. Loads key[s,:], cos[s,:], sin[s,:] once.
      2. Computes rotated key and RoPE → rope_out.
      3. Loads val[s,:] once.
      4. Writes rope_out → out_rope[0,0,s,:].
      5. Writes rope_out → out_expand[0,h,s,:] for h = 0..N_REP-1   (loop unrolled).
      6. Writes val       → out_val  [0,h,s,:] for h = 0..N_REP-1   (loop unrolled).

    Replaces 5 RoPE ops + 1 key expand+reshape + 1 val expand+reshape
    = 7 PyTorch kernel launches → 1 Triton kernel (3 CTAs).
    """
    s    = tl.program_id(0)
    d    = tl.arange(0, BLOCK_SIZE)
    half = BLOCK_SIZE // 2
    base = s * HEAD_DIM

    # ── RoPE on key ──────────────────────────────────────────────────────────
    key   = tl.load(key_ptr + base + d)
    cos_v = tl.load(cos_ptr + base + d)
    sin_v = tl.load(sin_ptr + base + d)

    is_first = d < half
    rot_d    = tl.where(is_first, d + half, d - half)
    rot_key  = tl.load(key_ptr + base + rot_d)
    rot_key  = tl.where(is_first, -rot_key, rot_key)

    rope_out = key * cos_v + rot_key * sin_v

    # Write rope output (once)
    tl.store(out_rope_ptr + base + d, rope_out)

    # ── Load value states ────────────────────────────────────────────────────
    val = tl.load(val_ptr + base + d)

    # ── Expand both key-rope and value for all N_REP heads (unrolled) ────────
    for h in range(N_REP):
        expand_base = h * total_seq * HEAD_DIM + s * HEAD_DIM
        tl.store(out_expand_ptr + expand_base + d, rope_out)
        tl.store(out_val_ptr    + expand_base + d, val)


# ── Pattern: match RoPE + key expand + value expand ───────────────────────────
def pattern(in_2, in_1, in_4, in_5):
    # RoPE computation on key states
    tmp_0 = in_2 * in_1
    tmp_1 = in_2[Ellipsis, slice(None, 128, None)]
    tmp_2 = in_2[Ellipsis, slice(128, None, None)]
    tmp_3 = -tmp_2
    tmp_4 = torch.cat((tmp_3, tmp_1), dim=-1)
    tmp_5 = tmp_4 * in_4
    tmp_6 = tmp_0 + tmp_5

    # Key expand + reshape
    tmp_7  = tmp_6[slice(None, None, None), slice(None, None, None), None, slice(None, None, None), slice(None, None, None)]
    tmp_8  = tmp_7.expand(1, 1, 8, 3, 256)
    tmp_9  = tmp_8.reshape(1, 8, 3, 256)

    # Value expand + reshape
    tmp_10 = in_5[slice(None, None, None), slice(None, None, None), None, slice(None, None, None), slice(None, None, None)]
    tmp_11 = tmp_10.expand(1, 1, 8, 3, 256)
    tmp_12 = tmp_11.reshape(1, 8, 3, 256)

    return tmp_6, tmp_9, tmp_12   # three observable outputs


def replacement_args(in_2, in_1, in_4, in_5):
    return (in_2, in_1, in_4, in_5)


# ── Inner opaque wrapper (contains Triton; invisible to torch.compile) ─────────
@torch.fx.wrap
def _rope_expand_val_kernel_call(key, cos, sin, val):
    # Shapes are fixed for this model: all inputs are [1, 1, 3, 256] bfloat16.
    # Constants hardcoded to minimize Python overhead at each call.
    out_rope   = torch.empty_like(key)                                                # [1, 1, 3, 256]
    out_expand = torch.empty(1, 8, 3, 256, dtype=key.dtype, device=key.device)       # [1, 8, 3, 256]
    out_val    = torch.empty(1, 8, 3, 256, dtype=val.dtype,  device=val.device)      # [1, 8, 3, 256]

    # Model inputs are always contiguous — skip .contiguous() checks.
    rope_expand_val_kernel[(3,)](          # grid = (total_seq,) = (3,)
        key, cos, sin, val,
        out_rope, out_expand, out_val,
        3,              # total_seq  (hardcoded)
        HEAD_DIM=256,   # HEAD_DIM   (hardcoded)
        N_REP=8,        # N_REP      (hardcoded)
        BLOCK_SIZE=256, # BLOCK_SIZE (hardcoded = HEAD_DIM)
    )

    return out_rope, out_expand, out_val


# ── Outer traceable wrapper (NO @torch.fx.wrap) ────────────────────────────────
# torch.fx traces into this, creating 3 getitem nodes so that
# len(copied_returning_nodes)==3 == len(match.returning_nodes)==3.
def rope_expand_val_replacement(key, cos, sin, val):
    _result    = _rope_expand_val_kernel_call(key, cos, sin, val)
    out_rope   = _result[0]   # getitem → replaces tmp_6
    out_expand = _result[1]   # getitem → replaces tmp_9
    out_val    = _result[2]   # getitem → replaces tmp_12
    return out_rope, out_expand, out_val


def replacement_func():
    return rope_expand_val_replacement