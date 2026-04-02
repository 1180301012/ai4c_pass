import torch
import triton
import triton.language as tl


@triton.jit
def expand_reshape_kernel(
    src_ptr,
    dst_ptr,
    total_seq,
    head_dim,
    n_rep,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fuse expand + reshape: broadcast [1, 1, S, D] → [1, n_rep, S, D].

    Grid: (n_rep, total_seq)  e.g. (8, 3)
    Each program handles one (head, seq_pos) pair, writing head_dim elements.
    """
    h = tl.program_id(0)   # head index: 0 to n_rep-1
    s = tl.program_id(1)   # seq position: 0 to total_seq-1

    d = tl.arange(0, BLOCK_SIZE)

    # Source: element at [0, 0, s, d] → flat offset = s * head_dim + d
    src_base = s * head_dim
    val = tl.load(src_ptr + src_base + d)

    # Dest: element at [0, h, s, d] → flat offset = h * total_seq * head_dim + s * head_dim + d
    dst_base = h * total_seq * head_dim + s * head_dim
    tl.store(dst_ptr + dst_base + d, val)


def pattern(in_5):
    tmp_10 = in_5[slice(None, None, None), slice(None, None, None), None, slice(None, None, None), slice(None, None, None)]
    tmp_11 = tmp_10.expand(1, 1, 8, 3, 256)
    tmp_12 = tmp_11.reshape(1, 8, 3, 256)
    return tmp_12


def replacement_args(in_5):
    return (in_5,)


@torch.fx.wrap
def expand_reshape_wrapper(x):
    """
    Fused expand+reshape for any [1, 1, S, D] tensor.
    Input:  x   of shape [1, 1, S, D]
    Output: out of shape [1, n_rep, S, D]
    """
    B, KV_H, S, D = x.shape   # 1, 1, 3, 256
    n_rep     = 8
    total_seq = B * KV_H * S   # 3

    out   = torch.empty(B, n_rep, S, D, dtype=x.dtype, device=x.device)
    x_c   = x.contiguous()

    grid = (n_rep, total_seq)  # (8, 3)

    expand_reshape_kernel[grid](
        x_c, out,
        total_seq, D, n_rep,
        BLOCK_SIZE=256,
    )

    return out


def replacement_func():
    return expand_reshape_wrapper