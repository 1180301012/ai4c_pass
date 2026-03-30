import math
import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Pattern: softmax(x, dim=2, _stacklevel=5) followed by unsqueeze(-1)
# This matches the tail of every target graph regardless of batch / spatial size.
# ---------------------------------------------------------------------------

def pattern(x):
    tmp_4 = torch.nn.functional.softmax(x, 2, _stacklevel=5)
    tmp_5 = tmp_4.unsqueeze(-1)
    return (tmp_5,)


def replacement_args(x):
    return (x,)


# ---------------------------------------------------------------------------
# Triton kernel: one program per row, computes softmax in float32 then
# stores in the original dtype.  BLOCK_S must be >= S (padded to power-of-2).
# ---------------------------------------------------------------------------

@triton.jit
def fused_softmax_kernel(
    input_ptr,
    output_ptr,
    N,
    S,
    stride_in_n,
    stride_out_n,
    BLOCK_S: tl.constexpr,
):
    row = tl.program_id(0)

    col_offsets = tl.arange(0, BLOCK_S)
    mask = col_offsets < S

    # Load one row of S elements; pad out-of-range with -inf so they don't
    # contribute to the maximum or the exponential sum.
    x = tl.load(
        input_ptr + row * stride_in_n + col_offsets,
        mask=mask,
        other=float("-inf"),
    )

    # ---- numerically-stable softmax in float32 ----
    x_f32 = x.to(tl.float32)
    x_max = tl.max(x_f32, axis=0)
    x_centered = x_f32 - x_max
    x_exp = tl.exp(x_centered)
    x_sum = tl.sum(x_exp, axis=0)
    x_softmax = (x_exp / x_sum).to(x.dtype)

    # Store – masked so we never write the padding elements.
    tl.store(
        output_ptr + row * stride_out_n + col_offsets,
        x_softmax,
        mask=mask,
    )


# ---------------------------------------------------------------------------
# Python wrapper (must be decorated with @torch.fx.wrap so FX treats it as
# a leaf and does not try to trace inside it).
# ---------------------------------------------------------------------------

@torch.fx.wrap
def fused_softmax_unsqueeze(x):
    """
    x   : shape [N, 1, S]   (produced by view() in every target graph)
    out : shape [N, 1, S, 1]  (softmax over dim=2, then unsqueeze(-1))
    """
    N = x.shape[0]   # batch size (middle dim is always 1)
    S = x.shape[2]   # spatial / sequence length

    # Flatten the leading dims to get a strictly 2-D [N, S] contiguous tensor.
    x_flat = x.contiguous().view(N, S)
    output_flat = torch.empty_like(x_flat)

    # BLOCK_S = next power-of-2 >= S  (required so tl.arange covers every element)
    BLOCK_S = 2 ** math.ceil(math.log2(max(S, 1)))

    # Heuristic: 1 warp per 32 elements, capped at 32 warps (1024 threads)
    num_warps = min(max(BLOCK_S // 32, 1), 32)

    fused_softmax_kernel[(N,)](
        x_flat,
        output_flat,
        N,
        S,
        x_flat.stride(0),
        output_flat.stride(0),
        BLOCK_S=BLOCK_S,
        num_warps=num_warps,
    )

    # Reshape back: [N, S] -> [N, 1, S, 1]  (view, no data copy)
    return output_flat.view(N, 1, S, 1)


# ---------------------------------------------------------------------------
# replacement_func: must be a zero-argument function returning a callable.
# ---------------------------------------------------------------------------

def replacement_func():
    return fused_softmax_unsqueeze