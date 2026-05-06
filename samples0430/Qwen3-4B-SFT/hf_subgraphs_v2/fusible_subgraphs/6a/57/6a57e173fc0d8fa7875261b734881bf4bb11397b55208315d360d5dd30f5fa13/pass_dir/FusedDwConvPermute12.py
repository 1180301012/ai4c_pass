import torch
import triton
import triton.language as tl


@triton.jit
def _perm_copy12_kernel(
    src_ptr,   # [B, C, L, H] contiguous input
    dst_ptr,   # [B, L, C, H] contiguous output
    B, C, L, H,
    IS_BF16: tl.constexpr,
):
    """
    Transpose-B dimension copy: dst[b,l,c,h] = src[b,c,l,h]
    Grid = B * C * L programs; each program copies H elements.
    """
    pid = tl.program_id(0)
    l = pid % L
    c = (pid // L) % C
    b = pid // (C * L)

    h = tl.arange(0, H)

    # src strides: [C*L*H, L*H, H, 1]
    src_off = b * (C * L * H) + c * (L * H) + l * H + h
    # dst strides: [L*C*H, C*H, H, 1]
    dst_off = b * (L * C * H) + l * (C * H) + c * H + h

    val = tl.load(src_ptr + src_off)
    tl.store(dst_ptr + dst_off, val)


# ── Pattern: fuse permute(0,2,1,3) + contiguous (groups=12 variant) ───────────
# Same single-arg pattern — the Triton kernel works for any (C,L,H) shape.

def pattern(in_1):
    """Match the permute+contiguous subgraph on the context layer (groups=12)."""
    perm = in_1.permute(0, 2, 1, 3)
    cont = perm.contiguous()
    return cont


def replacement_args(in_1):
    return (in_1,)


@torch.fx.wrap
def _fused_permute_contiguous12(in_1):
    """Replaces permute(0,2,1,3).contiguous() with a Triton copy-permute pass."""
    B, C, L, H = in_1.shape
    IS_BF16 = (in_1.dtype == torch.bfloat16)
    out = torch.empty((B, L, C, H), dtype=in_1.dtype, device=in_1.device)
    _perm_copy12_kernel[(B * C * L,)](
        in_1, out, B, C, L, H,
        IS_BF16=IS_BF16,
    )
    return out


def replacement_func():
    return _fused_permute_contiguous12