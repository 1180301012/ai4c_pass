import torch
import triton
import triton.language as tl


def pattern(in_2, in_1, in_4):
    tmp_0 = in_2 * in_1
    tmp_1 = in_2[Ellipsis, slice(None, 128, None)]
    tmp_2 = in_2[Ellipsis, slice(128, None, None)]
    tmp_3 = -tmp_2
    tmp_4 = torch.cat((tmp_3, tmp_1), dim=-1)
    tmp_5 = tmp_4 * in_4
    tmp_6 = tmp_0 + tmp_5
    tmp_7 = tmp_6[slice(None, None, None), slice(None, None, None), None, slice(None, None, None), slice(None, None, None)]
    tmp_8 = tmp_7.expand(1, 1, 8, 3, 256)
    tmp_9 = tmp_8.reshape(1, 8, 3, 256)
    return (tmp_6, tmp_9)


def replacement_args(in_2, in_1, in_4):
    return (in_2, in_1, in_4)


@triton.jit
def _rope_key_expand_kernel(
    k_ptr, cos_ptr, sin_ptr,
    k_embed_ptr, k_embed_exp_ptr,
    SEQ_LEN: tl.constexpr,
    DIM: tl.constexpr,
    N_HEADS: tl.constexpr,
):
    """
    One program per sequence position (SEQ_LEN=3 programs total).
    Each program handles one full row of DIM=256 elements.
    RoPE formula:
      embed[0:128]   = k[0:128]*cos[0:128]   - k[128:256]*sin[0:128]
      embed[128:256] = k[128:256]*cos[128:256] + k[0:128]*sin[128:256]
    Then broadcast embed to N_HEADS output heads.
    """
    row = tl.program_id(0)

    # Offsets for each half
    off1 = tl.arange(0, DIM // 2)           # 0 .. 127
    off2 = DIM // 2 + tl.arange(0, DIM // 2)  # 128 .. 255

    base = row * DIM

    # Load k, cos, sin for this row
    k1 = tl.load(k_ptr + base + off1)
    k2 = tl.load(k_ptr + base + off2)
    cos1 = tl.load(cos_ptr + base + off1)
    cos2 = tl.load(cos_ptr + base + off2)
    sin1 = tl.load(sin_ptr + base + off1)
    sin2 = tl.load(sin_ptr + base + off2)

    # RoPE computation
    embed1 = k1 * cos1 - k2 * sin1
    embed2 = k2 * cos2 + k1 * sin2

    # Store k_embed  [1,1,3,256] → flat offset = row*256
    tl.store(k_embed_ptr + base + off1, embed1)
    tl.store(k_embed_ptr + base + off2, embed2)

    # Store k_embed_exp [1,8,3,256] → flat offset = head*SEQ_LEN*DIM + row*DIM
    for h in range(N_HEADS):
        exp_base = h * SEQ_LEN * DIM + row * DIM
        tl.store(k_embed_exp_ptr + exp_base + off1, embed1)
        tl.store(k_embed_exp_ptr + exp_base + off2, embed2)


@torch.fx.wrap
def fused_rope_key_expand(k, cos, sin):
    """
    Inputs:
      k, cos, sin : [1, 1, 3, 256] bfloat16  (contiguous)
    Outputs:
      k_embed     : [1, 1, 3, 256] bfloat16
      k_embed_exp : [1, 8, 3, 256] bfloat16
    """
    SEQ_LEN = 3
    DIM = 256
    N_HEADS = 8

    k_embed = torch.empty_like(k)
    k_embed_exp = torch.empty(
        (1, N_HEADS, SEQ_LEN, DIM), dtype=k.dtype, device=k.device
    )

    _rope_key_expand_kernel[(SEQ_LEN,)](
        k, cos, sin,
        k_embed, k_embed_exp,
        SEQ_LEN=SEQ_LEN,
        DIM=DIM,
        N_HEADS=N_HEADS,
    )

    return k_embed, k_embed_exp


def replacement_func():
    return fused_rope_key_expand