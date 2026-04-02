"""
Optimization pass for the unfold+reshape pattern with C=16, H=8.

Input: in_0 [1, 16, seq_len]
Output: [seq_len*2, 8, 9]

The entire sequence of ops:
  contiguous -> unsqueeze(-1) -> unfold(kernel=[9,1], pad=[4,0]) ->
  transpose(1,2) -> reshape(1,-1,16,9) -> reshape([-1,8,9])

is equivalent to a direct sliding-window gather:
  out[n, h, k] = in_0[0, (n%2)*8 + h, n//2 + k - 4]  (zero-padded)
"""

import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Pattern: transpose + two reshapes (avoiding unfold, which the custom
# tracer can't trace as a single leaf node).
# in_0 matches tmp_2 (unfold output shape [1, C*9, seq_len]).
# Output: tmp_5 shape [seq_len*num_groups, head_size, 9]
# ---------------------------------------------------------------------------
def pattern(in_0):
    tmp_3 = in_0.transpose(1, 2)
    tmp_4 = tmp_3.reshape(1, -1, 16, 9)
    tmp_5 = torch.reshape(tmp_4, [-1, 8, 9])
    return (tmp_5,)


def replacement_args(in_0):
    return (in_0,)


# ---------------------------------------------------------------------------
# Triton kernel: fused transpose + reshape
# Input: tmp_2 [1, C*9, seq_len] = [1, 144, seq_len]
# Output: [seq_len*num_groups, head_size, 9]
# Mapping: out[n, h, k] = in[0, ((n%2)*8+h)*9+k, n//2]
# ---------------------------------------------------------------------------
@triton.autotune(
    configs=[
        triton.Config({'BLOCK': 64}),
        triton.Config({'BLOCK': 128}),
        triton.Config({'BLOCK': 256}),
        triton.Config({'BLOCK': 512}),
        triton.Config({'BLOCK': 1024}),
    ],
    key=['seq_len', 'n_total'],
)
@triton.jit
def _transpose_reshape_c16_h8_kernel(
    input_ptr,
    output_ptr,
    seq_len,
    n_total,
    HEAD_SIZE:  tl.constexpr,   # 8
    WINDOW:     tl.constexpr,   # 9
    NUM_GROUPS: tl.constexpr,   # 2
    BLOCK:      tl.constexpr,
):
    pid  = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_total

    k     = offs % WINDOW
    tmp_q = offs // WINDOW
    h     = tmp_q % HEAD_SIZE
    n     = tmp_q // HEAD_SIZE

    # n = l*NUM_GROUPS + g  → l = n//NUM_GROUPS, g = n%NUM_GROUPS
    l   = n // NUM_GROUPS
    g   = n % NUM_GROUPS
    c   = g * HEAD_SIZE + h

    # input[0, c*WINDOW+k, l] → offset in [1, C*WINDOW, seq_len]
    in_idx = (c * WINDOW + k) * seq_len + l

    val = tl.load(input_ptr + in_idx, mask=mask, other=0.0)
    tl.store(output_ptr + offs, val, mask=mask)


@torch.fx.wrap
def unfold_reshape_c16_h8(in_0):
    # in_0: tmp_2 = unfold output, shape [1, C*9, seq_len]
    B, CW, seq_len = in_0.shape   # CW = 144
    HEAD_SIZE  = 8
    WINDOW     = 9
    NUM_GROUPS = 2                # C/HEAD_SIZE = 16/8
    N_out      = seq_len * NUM_GROUPS
    n_total    = N_out * HEAD_SIZE * WINDOW

    out = torch.empty((N_out, HEAD_SIZE, WINDOW),
                      dtype=in_0.dtype, device=in_0.device)

    grid = lambda meta: ((n_total + meta['BLOCK'] - 1) // meta['BLOCK'],)
    _transpose_reshape_c16_h8_kernel[grid](
        in_0.contiguous(), out,
        seq_len=seq_len,
        n_total=n_total,
        HEAD_SIZE=HEAD_SIZE,
        WINDOW=WINDOW,
        NUM_GROUPS=NUM_GROUPS,
    )
    return out  # tensor, not tuple


def replacement_func():
    return unfold_reshape_c16_h8