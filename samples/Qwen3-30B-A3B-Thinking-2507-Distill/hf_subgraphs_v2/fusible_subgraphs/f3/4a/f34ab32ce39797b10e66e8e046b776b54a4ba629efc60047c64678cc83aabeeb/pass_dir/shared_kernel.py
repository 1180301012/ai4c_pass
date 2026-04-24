import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_N': 16, 'BLOCK_IC': 32}, num_warps=1),
        triton.Config({'BLOCK_N': 16, 'BLOCK_IC': 64}, num_warps=2),
        triton.Config({'BLOCK_N': 16, 'BLOCK_IC': 128}, num_warps=4),
        triton.Config({'BLOCK_N': 16, 'BLOCK_IC': 256}, num_warps=4),
        triton.Config({'BLOCK_N': 16, 'BLOCK_IC': 512}, num_warps=8),
        triton.Config({'BLOCK_N': 32, 'BLOCK_IC': 32}, num_warps=1),
        triton.Config({'BLOCK_N': 32, 'BLOCK_IC': 64}, num_warps=2),
        triton.Config({'BLOCK_N': 32, 'BLOCK_IC': 128}, num_warps=4),
        triton.Config({'BLOCK_N': 32, 'BLOCK_IC': 256}, num_warps=4),
        triton.Config({'BLOCK_N': 32, 'BLOCK_IC': 512}, num_warps=8),
        triton.Config({'BLOCK_N': 64, 'BLOCK_IC': 32}, num_warps=2),
        triton.Config({'BLOCK_N': 64, 'BLOCK_IC': 128}, num_warps=4),
        triton.Config({'BLOCK_N': 64, 'BLOCK_IC': 256}, num_warps=4),
        triton.Config({'BLOCK_N': 64, 'BLOCK_IC': 512}, num_warps=8),
        triton.Config({'BLOCK_N': 128, 'BLOCK_IC': 32}, num_warps=2),
        triton.Config({'BLOCK_N': 128, 'BLOCK_IC': 64}, num_warps=4),
        triton.Config({'BLOCK_N': 128, 'BLOCK_IC': 128}, num_warps=4),
        triton.Config({'BLOCK_N': 128, 'BLOCK_IC': 256}, num_warps=4),
        triton.Config({'BLOCK_N': 128, 'BLOCK_IC': 512}, num_warps=8),
        triton.Config({'BLOCK_N': 128, 'BLOCK_IC': 512}, num_warps=16),
    ],
    key=['N', 'IC'],
)
@triton.jit
def _fused_kernel(
    x_ptr, w_ptr, bias_ptr, ln_w_ptr, ln_b_ptr, out_ptr,
    B, N, IC,
    eps,
    BLOCK_N: tl.constexpr,
    BLOCK_IC: tl.constexpr,
):
    """
    Fused: 1x1 Conv2d (as matmul) + LayerNorm (over N channels) + ReLU
    Input shapes:
      x      : [B, IC, 1, 1]  (contiguous -> treat as [B, IC])
      w      : [N, IC, 1, 1]  (contiguous -> treat as [N, IC])
      bias   : [N]
      ln_w   : [N, 1, 1]  -> treat as [N]
      ln_b   : [N]
      output : [B, N, 1, 1]  (contiguous -> treat as [B, N])
    Grid: (B,)
    """
    pid_b = tl.program_id(0)

    n_offsets = tl.arange(0, BLOCK_N)
    mask_n = n_offsets < N

    # Accumulate dot product: acc[n] = sum_k(w[n,k] * x[b,k])
    acc = tl.zeros([BLOCK_N], dtype=tl.float32)
    for ic_start in range(0, IC, BLOCK_IC):
        ic_offsets = ic_start + tl.arange(0, BLOCK_IC)
        mask_ic = ic_offsets < IC

        # x[b, ic] -> offset = b*IC + ic
        x_vals = tl.load(x_ptr + pid_b * IC + ic_offsets,
                         mask=mask_ic, other=0.0).to(tl.float32)

        # w[n, ic] -> offset = n*IC + ic
        w_vals = tl.load(w_ptr + n_offsets[:, None] * IC + ic_offsets[None, :],
                         mask=mask_n[:, None] & mask_ic[None, :], other=0.0).to(tl.float32)

        acc += tl.sum(w_vals * x_vals[None, :], axis=1)

    # Add conv bias
    bias_vals = tl.load(bias_ptr + n_offsets, mask=mask_n, other=0.0).to(tl.float32)
    acc += bias_vals

    # Layer norm over N channels (spatial dims are 1x1)
    mean = tl.sum(acc, axis=0) / N
    diff = tl.where(mask_n, acc - mean, 0.0)
    var = tl.sum(diff * diff, axis=0) / N
    inv_std = 1.0 / tl.sqrt(var + eps)
    xn = diff * inv_std

    # Affine: ln_weight * xn + ln_bias
    ln_w = tl.load(ln_w_ptr + n_offsets, mask=mask_n, other=1.0).to(tl.float32)
    ln_b = tl.load(ln_b_ptr + n_offsets, mask=mask_n, other=0.0).to(tl.float32)
    xn = xn * ln_w + ln_b

    # ReLU
    y = tl.maximum(xn, 0.0)

    # Store: auto-cast float32 result to output dtype
    tl.store(out_ptr + pid_b * N + n_offsets, y, mask=mask_n)


@torch.fx.wrap
def dispatch_fused_conv_ln_relu(bias, w, ln_bias, ln_weight, x, route):
    """
    Shared dispatch wrapper for all channel variants.
    route: "C16" | "C19" | "C38" | "C128"
    """
    B = x.shape[0]
    if route == "C16":
        N = 16
    elif route == "C19":
        N = 19
    elif route == "C38":
        N = 38
    else:  # C128
        N = 128
    IC = x.shape[1]

    out = torch.empty((B, N, 1, 1), dtype=x.dtype, device=x.device)

    _fused_kernel[(B,)](
        x, w, bias, ln_weight, ln_bias, out,
        B, N, IC,
        1e-5,
    )

    return out