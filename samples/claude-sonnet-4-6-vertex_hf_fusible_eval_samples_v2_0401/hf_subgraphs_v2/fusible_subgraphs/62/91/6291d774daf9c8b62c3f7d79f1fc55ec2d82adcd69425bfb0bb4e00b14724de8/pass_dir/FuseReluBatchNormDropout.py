import torch
import triton
import triton.language as tl


def pattern(in_0, in_1, in_2, in_3, in_4):
    tmp_4 = torch.nn.functional.relu(in_4, inplace=False)
    tmp_5 = torch.nn.functional.batch_norm(tmp_4, in_0, in_1, in_3, in_2, False, 0.1, 1e-05)
    tmp_6 = torch.nn.functional.dropout(tmp_5, p=0.0, training=False)
    return tmp_6


def replacement_args(in_0, in_1, in_2, in_3, in_4):
    return (in_0, in_1, in_2, in_3, in_4)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 1},  num_warps=2),
        triton.Config({'BLOCK_M': 2},  num_warps=2),
        triton.Config({'BLOCK_M': 4},  num_warps=4),
        triton.Config({'BLOCK_M': 8},  num_warps=4),
        triton.Config({'BLOCK_M': 16}, num_warps=4),
        triton.Config({'BLOCK_M': 32}, num_warps=4),
        triton.Config({'BLOCK_M': 64}, num_warps=8),
        triton.Config({'BLOCK_M': 128}, num_warps=8),
    ],
    key=['N', 'C'],
)
@triton.jit
def _fused_relu_bn_kernel(
    x_ptr,
    mean_ptr,
    var_ptr,
    weight_ptr,
    bias_ptr,
    out_ptr,
    N,
    C,
    eps,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid = tl.program_id(0)
    row_start = pid * BLOCK_M

    col_offs = tl.arange(0, BLOCK_N)
    col_mask = col_offs < C

    # Load per-channel BN parameters and upcast to float32 for accuracy
    mean   = tl.load(mean_ptr   + col_offs, mask=col_mask, other=0.0).to(tl.float32)
    var    = tl.load(var_ptr    + col_offs, mask=col_mask, other=1.0).to(tl.float32)
    weight = tl.load(weight_ptr + col_offs, mask=col_mask, other=1.0).to(tl.float32)
    bias   = tl.load(bias_ptr   + col_offs, mask=col_mask, other=0.0).to(tl.float32)

    # Precompute per-channel affine params (avoids redundant work across rows)
    inv_std = 1.0 / tl.sqrt(var + eps)
    scale   = weight * inv_std          # [BLOCK_N]
    shift   = bias - mean * scale       # [BLOCK_N]

    row_offs = row_start + tl.arange(0, BLOCK_M)
    row_mask = row_offs < N

    # Load a [BLOCK_M, BLOCK_N] tile of the input
    x = tl.load(
        x_ptr + row_offs[:, None] * C + col_offs[None, :],
        mask=row_mask[:, None] & col_mask[None, :],
        other=0.0,
    ).to(tl.float32)

    # Fused ReLU + BatchNorm (inference)
    x   = tl.maximum(x, 0.0)
    out = x * scale[None, :] + shift[None, :]

    # Store – Triton automatically narrows float32 → output element dtype
    tl.store(
        out_ptr + row_offs[:, None] * C + col_offs[None, :],
        out,
        mask=row_mask[:, None] & col_mask[None, :],
    )


@torch.fx.wrap
def fused_relu_bn_dropout(in_0, in_1, in_2, in_3, in_4):
    """Fused kernel: ReLU + BatchNorm(inference) + Dropout(p=0, eval) → identity drop."""
    assert in_4.ndim == 2, f"Expected 2D input, got {in_4.ndim}D"
    N, C = in_4.shape
    BLOCK_N = triton.next_power_of_2(C)

    device = in_4.device
    # Move CPU BN buffers/params to the same device as the activation
    mean   = in_0.to(device=device)
    var    = in_1.to(device=device)
    bias   = in_2.to(device=device)
    weight = in_3.to(device=device)

    x   = in_4.contiguous()
    out = torch.empty_like(x)

    grid = lambda meta: (triton.cdiv(N, meta['BLOCK_M']),)

    _fused_relu_bn_kernel[grid](
        x, mean, var, weight, bias, out,
        N, C, 1e-05,
        BLOCK_N=BLOCK_N,
    )

    return out


def replacement_func():
    return fused_relu_bn_dropout