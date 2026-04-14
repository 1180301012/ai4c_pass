import torch
import triton
import triton.language as tl


def pattern(in_5):
    tmp_10 = in_5[slice(None, None, None), slice(None, None, None), None, slice(None, None, None), slice(None, None, None)]
    tmp_11 = tmp_10.expand(1, 1, 8, 3, 256)
    tmp_12 = tmp_11.reshape(1, 8, 3, 256)
    return tmp_12


def replacement_args(in_5):
    return (in_5,)


@triton.jit
def _expand_heads_kernel(
    v_ptr, v_exp_ptr,
    SEQ_LEN: tl.constexpr,
    DIM: tl.constexpr,
    N_HEADS: tl.constexpr,
):
    """
    One program per sequence position.
    Copies each row from v [1,1,3,256] to N_HEADS copies in v_exp [1,8,3,256].
    """
    row = tl.program_id(0)
    offsets = tl.arange(0, DIM)

    base = row * DIM
    v = tl.load(v_ptr + base + offsets)

    for h in range(N_HEADS):
        exp_base = h * SEQ_LEN * DIM + row * DIM
        tl.store(v_exp_ptr + exp_base + offsets, v)


@torch.fx.wrap
def expand_value_heads(v):
    """
    Input:  v      [1, 1, 3, 256] bfloat16  (contiguous)
    Output: v_exp  [1, 8, 3, 256] bfloat16
    """
    SEQ_LEN = 3
    DIM = 256
    N_HEADS = 8

    v_exp = torch.empty(
        (1, N_HEADS, SEQ_LEN, DIM), dtype=v.dtype, device=v.device
    )

    _expand_heads_kernel[(SEQ_LEN,)](
        v, v_exp,
        SEQ_LEN=SEQ_LEN,
        DIM=DIM,
        N_HEADS=N_HEADS,
    )

    return v_exp


def replacement_func():
    return expand_value_heads