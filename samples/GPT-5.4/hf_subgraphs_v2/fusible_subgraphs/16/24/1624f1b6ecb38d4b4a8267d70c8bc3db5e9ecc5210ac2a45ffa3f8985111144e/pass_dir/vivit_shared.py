import torch
import triton
import triton.language as tl

from graph_net_bench.torch.posion_dispatch_tensor import unwrap_tensor


@triton.jit
def _patch_embed_post_kernel(
    conv_ptr,
    cls_ptr,
    pos_ptr,
    out_ptr,
    BLOCK_T: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    TOKENS = 980
    OUT_TOKENS = 981
    HIDDEN = 768

    pid_t = tl.program_id(0)
    pid_h = tl.program_id(1)

    tok = pid_t * BLOCK_T + tl.arange(0, BLOCK_T)
    hid = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)

    tok_mask = tok < OUT_TOKENS
    hid_mask = hid < HIDDEN
    out_mask = tok_mask[:, None] & hid_mask[None, :]

    pos_vals = tl.load(
        pos_ptr + tok[:, None] * HIDDEN + hid[None, :],
        mask=out_mask,
        other=0.0,
    ).to(tl.float32)

    cls_vals = tl.load(cls_ptr + hid, mask=hid_mask, other=0.0).to(tl.float32)

    src_tok = tok - 1
    conv_mask = hid_mask[:, None] & (src_tok[None, :] >= 0) & (src_tok[None, :] < TOKENS)
    conv_block = tl.load(
        conv_ptr + hid[:, None] * TOKENS + src_tok[None, :],
        mask=conv_mask,
        other=0.0,
    ).to(tl.float32)
    conv_vals = tl.trans(conv_block)

    base_vals = tl.where(tok[:, None] == 0, cls_vals[None, :], conv_vals)
    out_vals = base_vals + pos_vals

    tl.store(
        out_ptr + tok[:, None] * HIDDEN + hid[None, :],
        out_vals,
        mask=out_mask,
    )


@triton.jit
def _layer_norm_768_kernel(
    x_ptr,
    gamma_ptr,
    beta_ptr,
    out_ptr,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    HIDDEN = 768

    row = tl.program_id(0)
    row_offs = row * HIDDEN

    sum_acc = tl.zeros((), dtype=tl.float32)
    sqsum_acc = tl.zeros((), dtype=tl.float32)

    for start in tl.static_range(0, 768, BLOCK_SIZE):
        cols = start + tl.arange(0, BLOCK_SIZE)
        mask = cols < 768
        x = tl.load(x_ptr + row_offs + cols, mask=mask, other=0.0).to(tl.float32)
        sum_acc += tl.sum(x, axis=0)
        sqsum_acc += tl.sum(x * x, axis=0)

    mean = sum_acc / HIDDEN
    var = sqsum_acc / HIDDEN - mean * mean
    var = tl.maximum(var, 0.0)
    inv_std = tl.rsqrt(var + eps)

    for start in tl.static_range(0, 768, BLOCK_SIZE):
        cols = start + tl.arange(0, BLOCK_SIZE)
        mask = cols < 768
        x = tl.load(x_ptr + row_offs + cols, mask=mask, other=0.0).to(tl.float32)
        gamma = tl.load(gamma_ptr + cols, mask=mask, other=1.0).to(tl.float32)
        beta = tl.load(beta_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        y = (x - mean) * inv_std
        y = y * gamma + beta
        tl.store(out_ptr + row_offs + cols, y, mask=mask)


@torch.fx.wrap
def vivit_dispatch(*args):
    route = args[-1]

    if route == "patch_embed_post":
        conv_out, cls_token, pos_embed, _ = args
        conv_out = unwrap_tensor(conv_out)
        cls_token = unwrap_tensor(cls_token)
        pos_embed = unwrap_tensor(pos_embed)

        out = torch.empty_like(pos_embed)
        grid = (triton.cdiv(981, 16), triton.cdiv(768, 128))
        _patch_embed_post_kernel[grid](
            conv_out,
            cls_token,
            pos_embed,
            out,
            BLOCK_T=16,
            BLOCK_H=128,
            num_warps=4,
            num_stages=2,
        )
        return out

    if route == "layer_norm_768":
        x, ln_weight, ln_bias, _ = args
        x = unwrap_tensor(x)
        ln_weight = unwrap_tensor(ln_weight)
        ln_bias = unwrap_tensor(ln_bias)

        out = torch.empty_like(x)
        rows = x.numel() // 768
        _layer_norm_768_kernel[(rows,)](
            x,
            ln_weight,
            ln_bias,
            out,
            1e-6,
            BLOCK_SIZE=256,
            num_warps=4,
            num_stages=2,
        )
        return out

    raise RuntimeError(f"Unknown route: {route}")


def shared_replacement_func():
    return vivit_dispatch