import torch
import triton
import triton.language as tl


def pattern(bias, weight, x):
    conv = torch.conv2d(x, weight, bias, (1, 1), (0, 0), (1, 1), 1)
    v = conv.view(4, 1, 192)
    sm = torch.nn.functional.softmax(v, 2, _stacklevel=5)
    out = sm.unsqueeze(-1)
    return (out,)


def replacement_args(bias, weight, x):
    return (bias, weight, x)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_N': 48},  num_warps=1),
        triton.Config({'BLOCK_N': 64},  num_warps=2),
        triton.Config({'BLOCK_N': 128}, num_warps=2),
        triton.Config({'BLOCK_N': 256}, num_warps=4),
        triton.Config({'BLOCK_N': 512}, num_warps=4),
        triton.Config({'BLOCK_N': 1024}, num_warps=8),
        triton.Config({'BLOCK_N': 2048}, num_warps=8),
        triton.Config({'BLOCK_N': 4096}, num_warps=16),
        triton.Config({'BLOCK_N': 8192}, num_warps=16),
    ],
    key=['N'],
)
@triton.jit
def _fused_view_softmax_kernel(
    x_ptr,       # [B, C, N] contiguous
    w_ptr,       # [C] (weight squeezed from [1,C,1,1])
    b_ptr,       # [1] (bias)
    out_ptr,     # [B, 1, N, 1] contiguous
    N,           # spatial flattened size (H*W)
    C,           # input channels
    BLOCK_N: tl.constexpr,
):
    """
    Fused kernel: view([B,1,H,W] -> [B,1,N]) + softmax(N) + unsqueeze -> [B,1,N,1].
    Each program handles one (batch, spatial) row.
    row_id = b * N + n
    """
    row_id = tl.program_id(0)
    b = row_id // N
    n_start = (row_id % N) * BLOCK_N

    # Load bias (scalar broadcast)
    bias_val = tl.load(b_ptr)

    n_offs = tl.arange(0, BLOCK_N)
    n = n_start + n_offs
    n_mask = n < N

    # Load weight slice [BLOCK_N] — weight[c] is at offset c
    # (weight has shape [1,C,1,1], so w_ptr + c gives weight[0,c,0,0])
    w = tl.load(w_ptr + tl.arange(0, C))   # [C]

    # Load x[b, :, n] as [C, BLOCK_N] with stride N between channels
    x = tl.load(
        x_ptr + b * C * N + tl.arange(0, C)[:, None] * N + n[None, :],
        mask=n_mask[None, :],
        other=-float('inf'),
    )  # [C, BLOCK_N]

    # Weighted sum: out[n] = bias + w @ x[:, n]
    val = bias_val + tl.sum(w[:, None] * x, axis=0)  # [BLOCK_N]

    # Numerically stable softmax over n
    val_max = tl.max(val, axis=0)
    val     = val - val_max
    exp_v   = tl.exp(val)
    sum_e   = tl.sum(exp_v, axis=0)
    softmax_out = exp_v / sum_e  # [BLOCK_N]

    # Store into [B, 1, N, 1]: element [b,0,n,0] is at b*N + n
    tl.store(out_ptr + b * N + n, softmax_out, mask=n_mask)


@torch.fx.wrap
def fused_conv1x1_view_softmax_unsqueeze(bias, weight, x):
    """
    Replacement for: conv2d(x,weight,bias) -> view(B,1,N) -> softmax(2) -> unsqueeze(-1)
    Returns a tuple containing the fused result [B, 1, N, 1].
    """
    B  = x.shape[0]
    C  = x.shape[1]
    H  = x.shape[2]
    W  = x.shape[3]
    N  = H * W

    # Squeeze weight [1, C, 1, 1] -> [C]
    w = weight.view(C)

    # Output shape [B, 1, N, 1]
    out = torch.empty((B, 1, N, 1), dtype=x.dtype, device=x.device)

    # One program per (b, n) row; BLOCK_N handled by autotune
    grid = lambda meta: (B * N,)

    _fused_view_softmax_kernel[grid](
        x, w, bias, out,
        N, C,
    )

    return (out,)


def replacement_func():
    return fused_conv1x1_view_softmax_unsqueeze