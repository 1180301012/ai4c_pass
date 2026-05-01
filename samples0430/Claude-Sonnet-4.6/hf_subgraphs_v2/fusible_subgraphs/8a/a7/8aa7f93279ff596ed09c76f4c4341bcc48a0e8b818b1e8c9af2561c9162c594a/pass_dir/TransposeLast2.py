import torch
import triton
import triton.language as tl

# -----------------------------------------------------------------------
# Pass 2: transpose last two dimensions  x.transpose(-2, -1)
#   Input shape:  [B0, B1, M, N]
#   Output shape: [B0, B1, N, M]
#
#   Strategy: flat-indexed read (coalesced) + index-computed scattered write.
# -----------------------------------------------------------------------

def pattern(x):
    return x.transpose(-2, -1)


def replacement_args(x):
    return (x,)


@triton.jit
def _transpose_last2_kernel(
    x_ptr, out_ptr,
    B, M, N,
    stride_xb, stride_xm, stride_xn,
    stride_ob, stride_on, stride_om,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Transposes the last two dimensions of a 3-D (batch, M, N) view.

    For each element (b, n, m) in the output [B, N, M]:
        out[b, n, m] = in[b, m, n]

    Each program handles BLOCK_SIZE contiguous output elements.
    """
    pid     = tl.program_id(0)
    offs    = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    n_total = B * N * M
    mask    = offs < n_total

    # Decode flat output index → (b, n, m)
    NM  = N * M
    b   = offs // NM
    rem = offs % NM
    n   = rem  // M
    m   = rem  % M

    # Read  in[b, m, n]  using input strides
    x = tl.load(
        x_ptr + b * stride_xb + m * stride_xm + n * stride_xn,
        mask=mask, other=0.0,
    )

    # Write  out[b, n, m]  using output strides
    tl.store(
        out_ptr + b * stride_ob + n * stride_on + m * stride_om,
        x,
        mask=mask,
    )


@torch.fx.wrap
def _transpose_last2(x):
    """
    Replaces x.transpose(-2, -1) for any 4-D tensor x = [B0, B1, M, N].
    Falls back gracefully to an arbitrary number of leading dims by
    merging everything except the last two dims into a single batch axis.
    """
    ndim = x.ndim
    # Merge all leading dims into a single batch
    B = 1
    for i in range(ndim - 2):
        B = B * x.shape[i]
    M = x.shape[-2]
    N = x.shape[-1]

    # Output shape: same leading dims but last two swapped
    out_shape = list(x.shape)
    out_shape[-2] = N
    out_shape[-1] = M
    out = torch.empty(out_shape, dtype=x.dtype, device=x.device)

    # Strides for the merged-batch axis: use stride of the first leading dim
    stride_xb = x.stride(0)  if ndim >= 1 else 1
    stride_xm = x.stride(-2)
    stride_xn = x.stride(-1)

    stride_ob = out.stride(0) if ndim >= 1 else 1
    stride_on = out.stride(-2)   # out dim[-2] = N  → stride per N step
    stride_om = out.stride(-1)   # out dim[-1] = M  → stride per M step

    n_out  = B * N * M
    BS     = 256
    grid   = ((n_out + BS - 1) // BS,)

    _transpose_last2_kernel[grid](
        x, out,
        B, M, N,
        stride_xb, stride_xm, stride_xn,
        stride_ob, stride_on, stride_om,
        BLOCK_SIZE=BS,
    )
    return out


def replacement_func():
    return _transpose_last2