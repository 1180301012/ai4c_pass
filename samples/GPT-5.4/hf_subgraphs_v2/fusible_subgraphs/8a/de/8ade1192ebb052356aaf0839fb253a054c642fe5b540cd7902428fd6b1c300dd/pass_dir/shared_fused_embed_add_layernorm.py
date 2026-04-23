import torch
import triton
import triton.language as tl


@triton.jit
def _fused_embed_add_layernorm_kernel(
    x_ptr,
    emb_ptr,
    gamma_ptr,
    beta_ptr,
    pos_ptr,
    out_ptr,
    n_rows,
    seq_len,
    H,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK_SIZE)
    cols_mask = offs < H

    seq_idx = row % seq_len
    pos = tl.load(pos_ptr + seq_idx)
    emb_row = pos + 2

    x_row_start = row * H
    emb_row_start = emb_row * H

    x = tl.load(x_ptr + x_row_start + offs, mask=cols_mask, other=0.0)
    e = tl.load(emb_ptr + emb_row_start + offs, mask=cols_mask, other=0.0)
    z = tl.cast(x, tl.float32) + tl.cast(e, tl.float32)

    mean = tl.sum(z, axis=0) / H
    centered = z - mean
    var = tl.sum(centered * centered, axis=0) / H
    inv_std = tl.rsqrt(var + eps)

    gamma = tl.load(gamma_ptr + offs, mask=cols_mask, other=1.0)
    beta = tl.load(beta_ptr + offs, mask=cols_mask, other=0.0)
    y = centered * inv_std
    y = y * tl.cast(gamma, tl.float32) + tl.cast(beta, tl.float32)

    tl.store(out_ptr + x_row_start + offs, y, mask=cols_mask)


def _launch_cfg(hidden):
    if hidden == 16:
        return 16, 1, 1
    if hidden == 768:
        return 1024, 4, 1
    if hidden == 1024:
        return 1024, 4, 1
    return triton.next_power_of_2(hidden), 4, 1


@torch.fx.wrap
def fused_embed_add_layernorm(in_0, in_1, in_2, in_3, in_4):
    out = torch.empty_like(in_0)

    hidden = in_0.shape[-1]
    seq_len = in_4.numel()
    n_rows = in_0.numel() // hidden
    if n_rows > 0:
        block_size, num_warps, num_stages = _launch_cfg(hidden)
        grid = (n_rows,)
        _fused_embed_add_layernorm_kernel[grid](
            in_0,
            in_1,
            in_3,
            in_2,
            in_4,
            out,
            n_rows,
            seq_len,
            hidden,
            1e-05,
            BLOCK_SIZE=block_size,
            num_warps=num_warps,
            num_stages=num_stages,
        )

    return out


def replacement_func():
    return fused_embed_add_layernorm