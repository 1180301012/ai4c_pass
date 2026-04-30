import torch
import triton
import triton.language as tl


def pattern(in_0, in_1):
    tmp_0 = in_1 + in_0
    tmp_1 = tmp_0.view(8, 300, 625)
    tmp_2 = torch.nn.functional.softmax(tmp_1, dim=-1)
    tmp_3 = tmp_2.view(1, 8, 300, 625)
    tmp_4 = tmp_3.view(8, 300, 625)
    tmp_5 = torch.nn.functional.dropout(tmp_4, p=0.0, training=False)
    return (tmp_5, tmp_3)


def replacement_args(in_0, in_1):
    return (in_0, in_1)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=16),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=32),
    ],
    key=['total_rows', 'seq_len'],
)
@triton.jit
def _fused_add_softmax_300(
    in0_ptr, in1_ptr, out_ptr,
    total_rows, seq_len,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fused add + softmax kernel.
    in_0: [1, 1, 300, 625]  – attention mask (broadcast over heads)
    in_1: [1, 8, 300, 625]  – main tensor
    out:  [1, 8, 300, 625]

    Each program handles one row of seq_len=625 elements.
    Grid = (8 * 300,) = 2400 programs.
    """
    row_idx = tl.program_id(0)
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < seq_len

    # Compute flat indices
    # in_1[0, h, s, :] flat index = h * seq_len + s * seq_len + col
    #                             = row_idx * seq_len + col
    # in_0[0, 0, s, :] flat index = (s * 625 + col) = row_idx * 625 + col
    in1_offs = row_idx * seq_len + col_offsets
    in0_offs = row_idx * seq_len + col_offsets  # same since H0=1, H1=8, same N*L layout

    # Load inputs, out-of-bounds → -inf for numerical stability of softmax
    in1_vals = tl.load(in1_ptr + in1_offs, mask=mask, other=float('-inf'))
    in0_vals = tl.load(in0_ptr + in0_offs, mask=mask, other=float('-inf'))

    # Add (accumulate in float32 for precision)
    x = in1_vals.to(tl.float32) + in0_vals.to(tl.float32)

    # Softmax: subtract row max for numerical stability
    x_max = tl.max(x, axis=0)
    x = x - x_max
    x_exp = tl.exp(x)
    x_sum = tl.sum(x_exp, axis=0)
    result = x_exp / x_sum  # float32

    # Store – Triton converts float32 → output dtype (bf16/fp16)
    out_offs = row_idx * seq_len + col_offsets
    tl.store(out_ptr + out_offs, result, mask=mask)


@torch.fx.wrap
def triton_fused_add_softmax_view_300(in_0, in_1):
    # in_0: [1, 1, 300, 625],  in_1: [1, 8, 300, 625]
    # out:  [1, 8, 300, 625]
    total_rows = 8 * 300   # 2400
    seq_len    = 625

    out = torch.empty((1, 8, 300, 625), dtype=in_1.dtype, device=in_1.device)

    _fused_add_softmax_300[(total_rows,)](
        in_0, in_1, out,
        total_rows, seq_len,
    )

    # tmp_3 and tmp_5 are the same tensor (dropout p=0.0 is identity)
    return (out, out)


def replacement_func():
    return triton_fused_add_softmax_view_300