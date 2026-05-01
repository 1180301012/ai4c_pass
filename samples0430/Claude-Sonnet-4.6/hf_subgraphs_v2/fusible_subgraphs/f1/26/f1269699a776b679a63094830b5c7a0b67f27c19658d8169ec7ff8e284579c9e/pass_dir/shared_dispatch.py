import torch
import triton
import triton.language as tl


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
def _eal_k1024(
    wids, pids, wemb, pemb, lnw, lnb, out,
    nt, H: tl.constexpr, eps: tl.constexpr, BH: tl.constexpr, DT: tl.constexpr,
):
    pid = tl.program_id(0)
    wi = tl.load(wids + pid)
    pi = tl.load(pids + pid)
    h  = tl.arange(0, BH)
    mk = h < H
    we = tl.load(wemb + wi * H + h, mask=mk, other=0.0).to(tl.float32)
    pe = tl.load(pemb + pi * H + h, mask=mk, other=0.0).to(tl.float32)
    x  = we + pe
    mn = tl.sum(tl.where(mk, x, 0.0), 0) / H
    df = tl.where(mk, x - mn, 0.0)
    vr = tl.sum(df * df, 0) / H
    xn = (x - mn) / tl.sqrt(vr + eps)
    lw = tl.load(lnw + h, mask=mk, other=0.0).to(tl.float32)
    lb = tl.load(lnb + h, mask=mk, other=0.0).to(tl.float32)
    o  = xn * lw + lb
    if DT == 2:
        o = o.to(tl.bfloat16)
    elif DT == 1:
        o = o.to(tl.float16)
    tl.store(out + pid * H + h, o, mask=mk)


@triton.autotune(
    configs=[
        triton.Config({'num_warps': 1}),
        triton.Config({'num_warps': 2}),
        triton.Config({'num_warps': 4}),
    ],
    key=[],
)
@triton.jit
def _eal_k64(
    wids, pids, wemb, pemb, lnw, lnb, out,
    nt, H: tl.constexpr, eps: tl.constexpr, BH: tl.constexpr, DT: tl.constexpr,
):
    pid = tl.program_id(0)
    wi = tl.load(wids + pid)
    pi = tl.load(pids + pid)
    h  = tl.arange(0, BH)
    we = tl.load(wemb + wi * H + h).to(tl.float32)
    pe = tl.load(pemb + pi * H + h).to(tl.float32)
    x  = we + pe
    mn = tl.sum(x, 0) / H
    df = x - mn
    vr = tl.sum(df * df, 0) / H
    xn = df / tl.sqrt(vr + eps)
    lw = tl.load(lnw + h).to(tl.float32)
    lb = tl.load(lnb + h).to(tl.float32)
    o  = xn * lw + lb
    if DT == 2:
        o = o.to(tl.bfloat16)
    elif DT == 1:
        o = o.to(tl.float16)
    tl.store(out + pid * H + h, o)


@torch.fx.wrap
def _dispatch(in_0, in_1, in_2, in_3, in_4, in_5, route):
    """
    in_0: word_ids  [B, S] int64
    in_1: ln_bias   [H]
    in_2: ln_weight [H]
    in_3: pos_emb   [pos_vocab, H]
    in_4: word_emb  [word_vocab, H]
    in_5: pos_ids   [B, S] int64
    """
    B   = in_0.shape[0]
    S   = in_0.shape[1]
    nt  = B * S
    dev = in_4.device
    dt  = in_4.dtype
    did = 2 if dt == torch.bfloat16 else (1 if dt == torch.float16 else 0)
    if route == "768_1e5":
        H   = 768
        out = torch.empty((B, S, H), dtype=dt, device=dev)
        _eal_k1024[(nt,)](
            in_0, in_5, in_4, in_3, in_2, in_1, out,
            nt=nt, H=H, eps=1e-5, BH=1024, DT=did,
        )
        return out
    else:   # "64_1e12"
        H   = 64
        out = torch.empty((B, S, H), dtype=dt, device=dev)
        _eal_k64[(nt,)](
            in_0, in_5, in_4, in_3, in_2, in_1, out,
            nt=nt, H=H, eps=1e-12, BH=64, DT=did,
        )
        return out