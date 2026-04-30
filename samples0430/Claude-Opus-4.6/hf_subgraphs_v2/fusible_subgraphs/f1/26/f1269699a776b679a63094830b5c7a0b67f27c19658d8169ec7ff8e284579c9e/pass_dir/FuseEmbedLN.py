import torch
import triton
import triton.language as tl


def pattern(in_0, in_1, in_2, in_3, in_4, in_5, ns, eps):
    tmp_5 = torch.nn.functional.embedding(in_0, in_4, 1, None, 2.0, False, False)
    tmp_6 = torch.nn.functional.embedding(in_5, in_3, 1, None, 2.0, False, False)
    tmp_7 = tmp_5 + tmp_6
    tmp_8 = torch.nn.functional.layer_norm(tmp_7, ns, in_2, in_1, eps)
    tmp_9 = torch.nn.functional.dropout(tmp_8, 0.1, False, False)
    return tmp_9


def replacement_args(in_0, in_1, in_2, in_3, in_4, in_5, ns, eps):
    return (in_0, in_1, in_2, in_3, in_4, in_5, eps)


@triton.jit
def _fused_emb_ln_kernel(
    indices0_ptr, indices5_ptr,
    weight4_ptr, weight3_ptr,
    ln_weight_ptr, ln_bias_ptr,
    output_ptr,
    HIDDEN: tl.constexpr,
    BLOCK: tl.constexpr,
    EPS: tl.constexpr,
):
    row = tl.program_id(0)
    idx0 = tl.load(indices0_ptr + row)
    idx5 = tl.load(indices5_ptr + row)
    cols = tl.arange(0, BLOCK)
    mask = cols < HIDDEN
    e0 = tl.load(weight4_ptr + idx0 * HIDDEN + cols, mask=mask, other=0.0)
    e5 = tl.load(weight3_ptr + idx5 * HIDDEN + cols, mask=mask, other=0.0)
    x = (e0 + e5).to(tl.float32)
    mean = tl.sum(x, axis=0) / HIDDEN
    xc = x - mean
    xc = tl.where(mask, xc, 0.0)
    var = tl.sum(xc * xc, axis=0) / HIDDEN
    rstd = 1.0 / tl.sqrt(var + EPS)
    xn = xc * rstd
    w = tl.load(ln_weight_ptr + cols, mask=mask, other=1.0).to(tl.float32)
    b = tl.load(ln_bias_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    out = xn * w + b
    tl.store(output_ptr + row * HIDDEN + cols, out, mask=mask)


@torch.fx.wrap
def fused_embed_ln(in_0, in_1, in_2, in_3, in_4, in_5, eps):
    B = in_0.shape[0]
    S = in_0.shape[1]
    H = in_4.shape[1]
    N = B * S
    BLOCK = 1 << (H - 1).bit_length()
    nw = 8 if BLOCK >= 512 else 2
    out = torch.empty(B, S, H, dtype=in_4.dtype, device=in_4.device)
    _fused_emb_ln_kernel[(N,)](
        in_0, in_5, in_4, in_3, in_2, in_1, out,
        H, BLOCK, eps, num_warps=nw,
    )
    return out


def replacement_func():
    return fused_embed_ln