"""
Optimization pass for the unfold+reshape pattern with C=384, H=64.

Input: in_0 [1, 384, seq_len]
Output: [seq_len*6, 64, 9]

The entire sequence of ops:
  contiguous -> unsqueeze(-1) -> unfold(kernel=[9,1], pad=[4,0]) ->
  transpose(1,2) -> reshape(1,-1,384,9) -> reshape([-1,64,9])

is equivalent to a direct sliding-window gather:
  out[n, h, k] = in_0[0, (n%6)*64 + h, n//6 + k - 4]  (zero-padded)
"""

import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Pattern to match – custom tracer treats torch.nn.functional.unfold as leaf
# ---------------------------------------------------------------------------
def pattern(in_0):
    tmp_0 = in_0.contiguous()
    tmp_1 = tmp_0.unsqueeze(-1)
    tmp_2 = torch.nn.functional.unfold(tmp_1, kernel_size=[9, 1], dilation=1, padding=[4, 0], stride=1)
    tmp_3 = tmp_2.transpose(1, 2)
    tmp_4 = tmp_3.reshape(1, -1, 384, 9)
    tmp_5 = torch.reshape(tmp_4, [-1, 64, 9])
    return (tmp_5,)


def replacement_args(in_0):
    return (in_0,)


# ---------------------------------------------------------------------------
# Triton kernel
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
def _unfold_reshape_c384_h64_kernel(
    input_ptr,
    output_ptr,
    seq_len,
    n_total,
    HEAD_SIZE:  tl.constexpr,   # 64
    WINDOW:     tl.constexpr,   # 9
    HALF_WIN:   tl.constexpr,   # 4
    NUM_GROUPS: tl.constexpr,   # 6  (= C // HEAD_SIZE = 384 // 64)
    BLOCK:      tl.constexpr,
):
    pid  = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_total

    # Decode linear index -> (n, h, k)
    k     = offs % WINDOW
    tmp_q = offs // WINDOW
    h     = tmp_q % HEAD_SIZE
    n     = tmp_q // HEAD_SIZE

    # Map n -> (seq_idx, group)
    seq_idx = n // NUM_GROUPS
    group   = n % NUM_GROUPS
    channel = group * HEAD_SIZE + h

    input_seq = seq_idx + k - HALF_WIN

    valid = (input_seq >= 0) & (input_seq < seq_len) & mask

    # Load from in_0[0, channel, input_seq]  (batch=1)
    val = tl.load(
        input_ptr + channel * seq_len + input_seq,
        mask=valid,
        other=0.0,
    )

    tl.store(output_ptr + offs, val, mask=mask)


# ---------------------------------------------------------------------------
# Wrapper
# ---------------------------------------------------------------------------
@torch.fx.wrap
def unfold_reshape_c384_h64(in_0):
    in_0 = in_0.contiguous()
    _, C, seq_len = in_0.shape          # C=384

    HEAD_SIZE  = 64
    WINDOW     = 9
    HALF_WIN   = 4
    NUM_GROUPS = C // HEAD_SIZE         # 6
    N_out      = seq_len * NUM_GROUPS
    n_total    = N_out * HEAD_SIZE * WINDOW

    out = torch.empty((N_out, HEAD_SIZE, WINDOW),
                      dtype=in_0.dtype, device=in_0.device)

    grid = lambda meta: ((n_total + meta['BLOCK'] - 1) // meta['BLOCK'],)
    _unfold_reshape_c384_h64_kernel[grid](
        in_0, out,
        seq_len=seq_len,
        n_total=n_total,
        HEAD_SIZE=HEAD_SIZE,
        WINDOW=WINDOW,
        HALF_WIN=HALF_WIN,
        NUM_GROUPS=NUM_GROUPS,
    )

    return out  # Return tensor, NOT tuple


def replacement_func():
    return unfold_reshape_c384_h64