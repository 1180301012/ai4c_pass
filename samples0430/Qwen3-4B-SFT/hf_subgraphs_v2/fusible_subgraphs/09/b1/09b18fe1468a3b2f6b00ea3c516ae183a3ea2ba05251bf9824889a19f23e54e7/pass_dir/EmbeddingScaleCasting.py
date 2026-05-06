import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Fused embedding-similarity kernel
# Computes: scale(in_3) * softmax(sum((in_1-in_2)^2, dim=3), dim=2)
# in_1 [1,N,C,D], in_2 [1,1,C,D], in_3 [1,1,C]  -> tmp_4 [1,N,C]
# Each CTA handles one n-row, iterates over D=512 channels
# ---------------------------------------------------------------------------

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_D': 512, 'BLOCK_N': 1}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_D': 512, 'BLOCK_N': 1}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_D': 512, 'BLOCK_N': 2}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_D': 512, 'BLOCK_N': 4}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_D': 256, 'BLOCK_N': 1}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_D': 256, 'BLOCK_N': 2}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_D': 256, 'BLOCK_N': 4}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_D': 512, 'BLOCK_N': 1}, num_warps=16, num_stages=3),
    ],
    key=['N', 'C', 'D'],
)
@triton.jit
def _embed_scale_kernel(
    in1_ptr,   # [1, N, C, D]
    in2_ptr,   # [1, 1, C, D]  (broadcast)
    in3_ptr,   # [C]
    out_ptr,   # [N, C]
    N, C, D,
    BLOCK_C: tl.constexpr,    # = 32
    BLOCK_N: tl.constexpr,    # rows per CTA
    BLOCK_D: tl.constexpr,    # D-tile width
):
    pid_n = tl.program_id(0)   # which n block
    pid_c = tl.program_id(1)   # which C block
    pid_d = tl.program_id(2)   # which D block

    n_start = pid_n * BLOCK_N
    c_start = pid_c * BLOCK_C
    d_start = pid_d * BLOCK_D

    n_off = n_start + tl.arange(0, BLOCK_N)   # [BLOCK_N]
    c_off = c_start + tl.arange(0, BLOCK_C)   # [BLOCK_C]
    d_off = d_start + tl.arange(0, BLOCK_D)   # [BLOCK_D]
    n_mask = n_off < N
    d_mask = d_off < D

    # sum over D of (in1[n,c,d] - in2[c,d])^2  for each c in this slice
    # in1 shape [N,C,D]... but in_1 is [1,N,C,D], batch-dim=0
    total = tl.zeros([BLOCK_N, BLOCK_C], dtype=tl.float32)

    for c in range(BLOCK_C):
        abs_c = c_start + c
        # Load in_1[0, n_off, abs_c, d_off]: [BLOCK_N, BLOCK_D]
        v1 = tl.load(in1_ptr + n_off[:, None] * C * D + abs_c * D + d_off[None, :],
                     mask=n_mask[:, None] & d_mask[None, :], other=0.0).to(tl.float32)
        # Load in_2[c, d]: [BLOCK_D] — broadcast over n
        v2 = tl.load(in2_ptr + abs_c * D + d_off, mask=d_mask, other=0.0).to(tl.float32)
        # Broadcast v2 from [BLOCK_D] to [BLOCK_N, BLOCK_D]
        v1_c = v1 - tl.broadcast_to(v2[None, :], (BLOCK_N, BLOCK_D))
        sq = tl.sum(v1_c * v1_c, axis=1)                      # [BLOCK_N]
        total += tl.where(d_mask[None, :], sq, 0.0)

    sum_sq = tl.sum(total, axis=1)                            # [BLOCK_N]

    # Per-channel scale from in_3[c]: [BLOCK_C]
    scale = tl.load(in3_ptr + c_off, mask=c_off < C, other=0.0).to(tl.float32)

    result = tl.broadcast_to(sum_sq[:, None], (BLOCK_N, BLOCK_C)) * scale[None, :]

    # Store: out[0, n, c]  — n_offsets * C + c_offsets (one n-row slice)
    out_ptrs = out_ptr + n_off[:, None] * C + c_off[None, :]
    tl.store(out_ptrs, result.to(in3_ptr.dtype.element_ty),
             mask=n_mask[:, None] & c_off[None, :] < C)


@torch.fx.wrap
def embed_scale_wrapper(in_1, in_2, in_3):
    """Returns tmp_4 = scale(in_3) * cost in float16/bfloat16 [1, N, C]"""
    N = in_1.shape[1]   # 4096
    C = in_1.shape[2]   # 32
    D = in_1.shape[3]   # 512
    out = torch.empty((1, N, C), dtype=in_1.dtype, device=in_1.device)

    def grid(meta):
        bN, bC, bD = meta['BLOCK_N'], meta['BLOCK_C'], meta['BLOCK_D']
        return (N // bN, C // bC, D // bD)

    _embed_scale_kernel[grid](
        in_1, in_2, in_3, out,
        N, C, D,
        BLOCK_C=32,
    )
    return out


@torch.fx.wrap
def embed_scale_wrapper(in_1, in_2, in_3):
    """Returns tmp_4 = scale(in_3) * cost in float16/bfloat16 [1, N, C]"""
    N = in_1.shape[1]   # 4096
    C = in_1.shape[2]   # 32
    D = in_1.shape[3]   # 512

    out = torch.empty((1, N, C), dtype=in_1.dtype, device=in_1.device)

    def grid(meta):
        return (
            (N + meta['BLOCK_N'] - 1) // meta['BLOCK_N'],
            C // meta['BLOCK_C'],
            D // meta['BLOCK_D'],
        )

    _embed_scale_kernel[grid](
        in_1, in_2, in_3, out,
        N, C, D,
        BLOCK_C=32,
    )
    return out


# ---------------------------------------------------------------------------
# Pattern / replacement API
# Matches the scale-costing computation chain:
#   (in_1 - in_2)^2, sum(dim=3), in_3 * sum   -> tmp_4
# in_1, in_2, in_3 are all direct model placeholders.
# The result tmp_4 feeds into the downstream softmax/unsqueeze.
# ---------------------------------------------------------------------------

def pattern(in_1, in_2, in_3):
    tmp_1 = in_1 - in_2
    tmp_2 = tmp_1.pow(2)
    tmp_3 = tmp_2.sum(dim=3)
    tmp_4 = in_3 * tmp_3
    return tmp_4


def replacement_args(in_1, in_2, in_3):
    return (in_1, in_2, in_3)


def replacement_func():
    return embed_scale_wrapper