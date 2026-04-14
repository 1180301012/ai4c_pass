import torch
import triton
import triton.language as tl


@triton.jit
def rope_q_kernel(
    x_ptr, cos_ptr, sin_ptr, cls_ptr, out_ptr,
    H, N, D,
    x_stride_h, x_stride_n,
    cos_stride_n, cls_stride_h,
    BLOCK_D: tl.constexpr,
):
    """
    RoPE + prepend cls for query.
    Grid: (H * (N+1),) — one program per (h, n_out) pair.
    n_out==0: copy cls token.  n_out>0: compute RoPE.
    """
    pid    = tl.program_id(0)
    h      = pid // (N + 1)
    n_out  = pid % (N + 1)

    d_idx  = tl.arange(0, BLOCK_D)
    out_base = (h * (N + 1) + n_out) * D

    if n_out == 0:
        cls_base = h * cls_stride_h
        val = tl.load(cls_ptr + cls_base + d_idx)
        tl.store(out_ptr + out_base + d_idx, val)
    else:
        n      = n_out - 1
        d_xor  = d_idx ^ 1
        even_m = (d_idx % 2) == 0

        x_base   = h * x_stride_h + n * x_stride_n
        emb_base = n * cos_stride_n

        x_val   = tl.load(x_ptr + x_base + d_idx)
        x_pair  = tl.load(x_ptr + x_base + d_xor)
        cos_val = tl.load(cos_ptr + emb_base + d_idx)
        sin_val = tl.load(sin_ptr + emb_base + d_idx)

        x_rot   = tl.where(even_m, -x_pair, x_pair)
        out_val = x_val * cos_val + x_rot * sin_val
        tl.store(out_ptr + out_base + d_idx, out_val)


@triton.jit
def rope_k_kernel(
    k_ptr, posembed_ptr, out_ptr,
    H, N, D,
    k_stride_h, k_stride_n,
    posembed_stride_n,
    BLOCK_D: tl.constexpr,
):
    """
    RoPE for key + keep cls token.
    Grid: (H * (N+1),) — one program per (h, n) pair.
    n==0: copy cls token.  n>0: compute RoPE.
    pos_embed layout: [N, 2D], sin = posembed[:, :D], cos = posembed[:, D:]
    """
    pid   = tl.program_id(0)
    h     = pid // (N + 1)
    n     = pid % (N + 1)

    d_idx    = tl.arange(0, BLOCK_D)
    out_base = (h * (N + 1) + n) * D

    if n == 0:
        k_cls_base = h * k_stride_h
        val = tl.load(k_ptr + k_cls_base + d_idx)
        tl.store(out_ptr + out_base + d_idx, val)
    else:
        pos    = n - 1
        d_xor  = d_idx ^ 1
        even_m = (d_idx % 2) == 0

        k_base   = h * k_stride_h + n * k_stride_n
        cos_base = pos * posembed_stride_n + D
        sin_base = pos * posembed_stride_n

        k_val   = tl.load(k_ptr   + k_base   + d_idx)
        k_pair  = tl.load(k_ptr   + k_base   + d_xor)
        cos_val = tl.load(posembed_ptr + cos_base + d_idx)
        sin_val = tl.load(posembed_ptr + sin_base + d_idx)

        k_rot   = tl.where(even_m, -k_pair, k_pair)
        out_val = k_val * cos_val + k_rot * sin_val
        tl.store(out_ptr + out_base + d_idx, out_val)


def _rope_q_impl(x, cos, sin, cls, ref):
    B, H, N, D = x.shape
    out = torch.empty(B, H, N + 1, D, dtype=x.dtype, device=x.device)
    nw  = 8 if H >= 12 else 4
    rope_q_kernel[(H * (N + 1),)](
        x, cos, sin, cls, out,
        H, N, D,
        x.stride(1), x.stride(2), cos.stride(0), cls.stride(1),
        BLOCK_D=64, num_warps=nw, num_stages=2,
    )
    return out


def _rope_k_impl(posembed, k, ref):
    B, H, Np1, D = k.shape
    N   = Np1 - 1
    out = torch.empty(B, H, Np1, D, dtype=k.dtype, device=k.device)
    nw  = 8 if H >= 12 else 4
    rope_k_kernel[(H * Np1,)](
        k, posembed, out,
        H, N, D,
        k.stride(1), k.stride(2), posembed.stride(0),
        BLOCK_D=64, num_warps=nw, num_stages=2,
    )
    return out


@torch.fx.wrap
def dispatch_rope(a, b, c, d, e, route):
    """
    Unified dispatch for RoPE operations.
    route "rope_q": a=x, b=cos, c=sin, d=cls, e=ref
    route "rope_k": a=posembed, b=k, c=ref, d/e are dummy (=ref)
    """
    if route == "rope_q":
        return _rope_q_impl(a, b, c, d, e)
    elif route == "rope_k":
        return _rope_k_impl(a, b, c)
    return _rope_q_impl(a, b, c, d, e)