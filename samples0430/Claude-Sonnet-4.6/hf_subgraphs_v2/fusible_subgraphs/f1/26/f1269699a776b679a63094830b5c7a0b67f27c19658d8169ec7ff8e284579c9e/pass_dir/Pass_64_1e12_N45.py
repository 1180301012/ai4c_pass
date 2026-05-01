import math
import torch
import triton
import triton.language as tl

_RPB_7  = [[(16 if i<j else 0) + (abs(i-j) if abs(i-j)<8 else min(8+int(8.0*math.log(abs(i-j)/8.0)/math.log(16.0)),15)) for j in range(7)]  for i in range(7)]
_RPB_11 = [[(16 if i<j else 0) + (abs(i-j) if abs(i-j)<8 else min(8+int(8.0*math.log(abs(i-j)/8.0)/math.log(16.0)),15)) for j in range(11)] for i in range(11)]
_RPB_45 = [[(16 if i<j else 0) + (abs(i-j) if abs(i-j)<8 else min(8+int(8.0*math.log(abs(i-j)/8.0)/math.log(16.0)),15)) for j in range(45)] for i in range(45)]


@triton.autotune(
    configs=[
        triton.Config({'num_warps': 2}),
        triton.Config({'num_warps': 4}),
        triton.Config({'num_warps': 8}),
        triton.Config({'num_warps': 16}),
    ],
    key=[],
)
@triton.jit
def _eal_kernel_1024b(
    word_ids_ptr, pos_ids_ptr,
    word_emb_ptr, pos_emb_ptr,
    ln_w_ptr, ln_b_ptr,
    out_ptr,
    num_tokens,
    H: tl.constexpr,
    eps: tl.constexpr,
    BLOCK_H: tl.constexpr,
    DTYPE: tl.constexpr,
):
    pid = tl.program_id(0)
    word_idx = tl.load(word_ids_ptr + pid)
    pos_idx  = tl.load(pos_ids_ptr  + pid)
    h = tl.arange(0, BLOCK_H)
    mask = h < H
    we = tl.load(word_emb_ptr + word_idx * H + h, mask=mask, other=0.0).to(tl.float32)
    pe = tl.load(pos_emb_ptr  + pos_idx  * H + h, mask=mask, other=0.0).to(tl.float32)
    x  = we + pe
    mean = tl.sum(tl.where(mask, x, 0.0), axis=0) / H
    diff = tl.where(mask, x - mean, 0.0)
    var  = tl.sum(diff * diff, axis=0) / H
    xn   = (x - mean) / tl.sqrt(var + eps)
    lw = tl.load(ln_w_ptr + h, mask=mask, other=0.0).to(tl.float32)
    lb = tl.load(ln_b_ptr + h, mask=mask, other=0.0).to(tl.float32)
    out = xn * lw + lb
    if DTYPE == 2:
        out = out.to(tl.bfloat16)
    elif DTYPE == 1:
        out = out.to(tl.float16)
    tl.store(out_ptr + pid * H + h, out, mask=mask)


@triton.autotune(
    configs=[
        triton.Config({'num_warps': 1}),
        triton.Config({'num_warps': 2}),
        triton.Config({'num_warps': 4}),
        triton.Config({'num_warps': 8}),
    ],
    key=[],
)
@triton.jit
def _eal_kernel_64b(
    word_ids_ptr, pos_ids_ptr,
    word_emb_ptr, pos_emb_ptr,
    ln_w_ptr, ln_b_ptr,
    out_ptr,
    num_tokens,
    H: tl.constexpr,
    eps: tl.constexpr,
    BLOCK_H: tl.constexpr,
    DTYPE: tl.constexpr,
):
    pid = tl.program_id(0)
    word_idx = tl.load(word_ids_ptr + pid)
    pos_idx  = tl.load(pos_ids_ptr  + pid)
    h = tl.arange(0, BLOCK_H)
    we = tl.load(word_emb_ptr + word_idx * H + h).to(tl.float32)
    pe = tl.load(pos_emb_ptr  + pos_idx  * H + h).to(tl.float32)
    x  = we + pe
    mean = tl.sum(x, axis=0) / H
    diff = x - mean
    var  = tl.sum(diff * diff, axis=0) / H
    xn   = diff / tl.sqrt(var + eps)
    lw = tl.load(ln_w_ptr + h).to(tl.float32)
    lb = tl.load(ln_b_ptr + h).to(tl.float32)
    out = xn * lw + lb
    if DTYPE == 2:
        out = out.to(tl.bfloat16)
    elif DTYPE == 1:
        out = out.to(tl.float16)
    tl.store(out_ptr + pid * H + h, out)


@torch.fx.wrap
def _dispatch(in_0, in_1, in_2, in_3, in_4, in_5, route):
    B = in_0.shape[0]
    S = in_0.shape[1]
    nt = B * S
    dev = in_4.device
    dt  = in_4.dtype
    did = 2 if dt == torch.bfloat16 else (1 if dt == torch.float16 else 0)
    if route == "768_1e5_11":
        H = 768
        out = torch.empty((B, S, H), dtype=dt, device=dev)
        _eal_kernel_1024b[(nt,)](
            in_0, in_5, in_4, in_3, in_2, in_1, out,
            num_tokens=nt, H=H, eps=1e-5, BLOCK_H=1024, DTYPE=did,
        )
        return (out, torch.as_tensor(_RPB_11, dtype=torch.int64))
    elif route == "64_1e12_45":
        H = 64
        out = torch.empty((B, S, H), dtype=dt, device=dev)
        _eal_kernel_64b[(nt,)](
            in_0, in_5, in_4, in_3, in_2, in_1, out,
            num_tokens=nt, H=H, eps=1e-12, BLOCK_H=64, DTYPE=did,
        )
        return (out, torch.as_tensor(_RPB_45, dtype=torch.int64))
    else:
        H = 768
        out = torch.empty((B, S, H), dtype=dt, device=dev)
        _eal_kernel_1024b[(nt,)](
            in_0, in_5, in_4, in_3, in_2, in_1, out,
            num_tokens=nt, H=H, eps=1e-5, BLOCK_H=1024, DTYPE=did,
        )
        return (out, torch.as_tensor(_RPB_7, dtype=torch.int64))


# ── Pass: 64-dim, eps=1e-12, N=45 (tiny-MPNet bfloat16 / float16) ─────────────
def pattern(in_0, in_1, in_2, in_3, in_4, in_5):
    tmp_5 = torch.nn.functional.embedding(in_0, in_4, 1, None, 2.0, False, False)
    tmp_6 = torch.nn.functional.embedding(in_5, in_3, 1, None, 2.0, False, False)
    tmp_7 = tmp_5 + tmp_6
    tmp_8 = torch.nn.functional.layer_norm(tmp_7, (64,), in_2, in_1, 1e-12)
    tmp_9 = torch.nn.functional.dropout(tmp_8, 0.1, False, False)
    tmp_10 = torch.arange(45, dtype=torch.int64)
    tmp_11 = tmp_10[(slice(None, None, None), None)]
    tmp_12 = torch.arange(45, dtype=torch.int64)
    tmp_13 = tmp_12[(None, slice(None, None, None))]
    tmp_14 = tmp_13 - tmp_11
    tmp_15 = -tmp_14
    tmp_16 = tmp_15 < 0
    tmp_17 = tmp_16.to(torch.int64)
    tmp_18 = tmp_17 * 16
    tmp_19 = 0 + tmp_18
    tmp_20 = torch.abs(tmp_15)
    tmp_21 = tmp_20 < 8
    tmp_22 = tmp_20.float()
    tmp_23 = tmp_22 / 8
    tmp_24 = torch.log(tmp_23)
    tmp_25 = tmp_24 / 2.772588722239781
    tmp_26 = tmp_25 * 8
    tmp_27 = tmp_26.to(torch.int64)
    tmp_28 = 8 + tmp_27
    tmp_29 = torch.full_like(tmp_28, 15)
    tmp_30 = torch.min(tmp_28, tmp_29)
    tmp_31 = torch.where(tmp_21, tmp_20, tmp_30)
    tmp_19 += tmp_31
    tmp_32 = tmp_19
    return (tmp_9, tmp_32)


def replacement_args(in_0, in_1, in_2, in_3, in_4, in_5):
    return (in_0, in_1, in_2, in_3, in_4, in_5, "64_1e12_45")


def replacement_func():
    return _dispatch