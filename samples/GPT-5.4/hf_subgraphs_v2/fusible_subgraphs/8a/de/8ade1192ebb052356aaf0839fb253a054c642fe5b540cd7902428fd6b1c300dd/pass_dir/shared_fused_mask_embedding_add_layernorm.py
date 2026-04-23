import torch
import triton
import triton.language as tl


@triton.jit
def _layernorm_kernel(
    x_ptr,
    gamma_ptr,
    beta_ptr,
    out_ptr,
    n_rows,
    H,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK_SIZE)
    cols_mask = offs < H

    row_start = row * H
    x = tl.load(x_ptr + row_start + offs, mask=cols_mask, other=0.0)
    x = tl.cast(x, tl.float32)

    mean = tl.sum(x, axis=0) / H
    centered = x - mean
    var = tl.sum(centered * centered, axis=0) / H
    inv_std = tl.rsqrt(var + eps)

    gamma = tl.load(gamma_ptr + offs, mask=cols_mask, other=1.0)
    beta = tl.load(beta_ptr + offs, mask=cols_mask, other=0.0)
    y = centered * inv_std
    y = y * tl.cast(gamma, tl.float32) + tl.cast(beta, tl.float32)

    tl.store(out_ptr + row_start + offs, y, mask=cols_mask)


def _num_warps_for_hidden(hidden):
    if hidden >= 1024:
        return 8
    if hidden >= 512:
        return 4
    if hidden >= 128:
        return 2
    return 1


@torch.fx.wrap
def fused_layernorm(x, gamma, beta):
    out = torch.empty_like(x)

    hidden = x.shape[-1]
    n_rows = x.numel() // hidden
    if n_rows > 0:
        block_size = triton.next_power_of_2(hidden)
        num_warps = _num_warps_for_hidden(hidden)
        grid = (n_rows,)
        _layernorm_kernel[grid](
            x,
            gamma,
            beta,
            out,
            n_rows,
            hidden,
            1e-05,
            BLOCK_SIZE=block_size,
            num_warps=num_warps,
            num_stages=1,
        )

    return out


def replacement_func():
    return fused_layernorm