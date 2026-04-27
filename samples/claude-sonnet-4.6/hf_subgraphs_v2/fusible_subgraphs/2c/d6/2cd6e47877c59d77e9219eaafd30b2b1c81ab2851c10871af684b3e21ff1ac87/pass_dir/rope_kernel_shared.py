"""
Shared Triton kernels for fused RoPE (Rotary Position Embedding) computation.

Stride-aware: works for both contiguous Q-stream and non-contiguous K-stream inputs.
The Q-pattern structurally matches BOTH Q-stream and K-stream.
Two focused kernels (cls_copy + rope_body) — no autotune overhead.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def cls_copy_kernel(
    cls_ptr,        # CLS token tensor (may be strided)
    out_ptr,        # (1, H, S+1, D) contiguous output
    Sp1,
    cls_stride_h,   # stride for h-dim of cls
    D: tl.constexpr,
):
    """Grid: H. Copies cls[0,h,0,:] → out[0,h,0,:] using actual h-stride."""
    h = tl.program_id(0)
    d = tl.arange(0, D)
    val = tl.load(cls_ptr + h * cls_stride_h + d)
    tl.store(out_ptr + h * Sp1 * D + d, val)


@triton.jit
def rope_body_kernel(
    x_ptr,          # Sequence tensor (may be strided)
    cos_ptr,        # Cosine embedding (may be strided)
    sin_ptr,        # Sine embedding (may be strided)
    out_ptr,        # (1, H, S+1, D) contiguous
    S,
    Sp1,
    x_stride_h,     # stride for h-dim of x
    emb_stride_s,   # stride for s-dim of embeddings
    D: tl.constexpr,
):
    """Grid: H*S. out[h,s+1,:] = x[h,s,:]*cos[s,:] + rotate(x[h,s,:])*sin[s,:]."""
    pid = tl.program_id(0)
    h = pid // S
    s = pid % S

    d = tl.arange(0, D)
    x_base = x_ptr + h * x_stride_h + s * D
    x  = tl.load(x_base + d)
    c  = tl.load(cos_ptr + s * emb_stride_s + d)
    sv = tl.load(sin_ptr + s * emb_stride_s + d)

    is_even = ((d % 2) == 0)
    partner = tl.where(is_even, d + 1, d - 1)
    x_p     = tl.load(x_base + partner)
    rotated = tl.where(is_even, -x_p, x_p)
    result  = x * c + rotated * sv

    tl.store(out_ptr + h * Sp1 * D + (s + 1) * D + d, result)


@torch.fx.wrap
def rope_dispatch(*args):
    """
    Unified dispatch for all RoPE passes.
    Route "Q": args = (in_1, in_2, in_3, in_5, in_6)
      Q-stream: x=q (contiguous), cos=cos_emb, sin=sin_emb, cls=q_cls
      K-stream (matched via Q-pattern): x=k_seq (strided), cos/sin=pos_embed halves (strided)
    Stride-aware: uses actual tensor strides for correct memory access.
    """
    route = args[-1]

    if route == "Q":
        in_1, in_2, in_3, in_5, in_6 = args[0], args[1], args[2], args[3], args[4]
        _B, H, S, D = in_3.shape
        Sp1 = S + 1
        out = torch.empty((1, H, Sp1, D), dtype=in_6.dtype, device=in_3.device)

        cls_stride_h = in_2.stride(1)   # D for q_cls; Sp1*D for k_cls
        x_stride_h   = in_3.stride(1)   # S*D for q; Sp1*D for k_seq
        emb_stride_s = in_1.stride(0)   # D for cos/sin; 2*D for pos_embed halves

        cls_copy_kernel[H,](in_2, out, Sp1, cls_stride_h, D)
        rope_body_kernel[H * S,](in_3, in_1, in_5, out, S, Sp1, x_stride_h, emb_stride_s, D)
        return out

    else:  # route == "K" -- unreachable (Q-pattern handles K structurally)
        in_0, in_4, in_6 = args[0], args[1], args[2]
        _B, H, Sp1, D = in_4.shape
        out = torch.empty((1, H, Sp1, D), dtype=in_6.dtype, device=in_4.device)
        return out