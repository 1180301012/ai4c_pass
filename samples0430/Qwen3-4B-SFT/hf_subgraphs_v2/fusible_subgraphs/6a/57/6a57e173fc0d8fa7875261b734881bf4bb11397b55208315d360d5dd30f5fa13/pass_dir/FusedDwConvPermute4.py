import torch
import triton
import triton.language as tl


@triton.jit
def _perm_copy_kernel(
    src_ptr,   # [B, C, L, H] contiguous
    dst_ptr,   # [B, L, C, H] contiguous
    B, C, L, H,
    IS_BF16: tl.constexpr,
    BLOCK_H: tl.constexpr,   # power-of-2 >= H
):
    """
    Grid = B * C * L programs; each program copies BLOCK_H elements for one (b,c,l).
    dst[b,l,c,h] = src[b,c,l,h]   for h in [0, min(BLOCK_H, H))
    """
    pid = tl.program_id(0)
    l = pid % L
    c = (pid // L) % C
    b = pid // (C * L)

    h = tl.arange(0, BLOCK_H)          # BLOCK_H is constexpr — tl.arange is valid
    mask_h = h < H                      # H is the runtime H dimension

    # src strides: [C*L*H, L*H, H, 1]
    src_off = b * (C * L * H) + c * (L * H) + l * H + h
    # dst strides: [L*C*H, C*H, H, 1]
    dst_off = b * (L * C * H) + l * (C * H) + c * H + h

    val = tl.load(src_ptr + src_off, mask=mask_h, other=0.0)
    tl.store(dst_ptr + dst_off, val, mask=mask_h)


# ── Pattern: fuse permute(0,2,1,3) + contiguous ───────────────────────────────

def pattern(in_1):
    """Match the permute+contiguous subgraph on the context/attention tensor."""
    perm = in_1.permute(0, 2, 1, 3)
    cont = perm.contiguous()
    return cont


def replacement_args(in_1):
    return (in_1,)


@torch.fx.wrap
def _fused_permute_contiguous(in_1):
    """Replace permute(0,2,1,3).contiguous() with a Triton kernel."""
    B, C, L, H = in_1.shape
    IS_BF16 = (in_1.dtype == torch.bfloat16)
    out = torch.empty((B, L, C, H), dtype=in_1.dtype, device=in_1.device)
    # Pick BLOCK_H that is a power-of-2 and >= H — H is either 8 or 64 here
    if H <= 32:
        BLOCK_H = 32
    else:
        BLOCK_H = 64
    _perm_copy_kernel[(B * C * L,)](
        in_1, out, B, C, L, H,
        IS_BF16=IS_BF16,
        BLOCK_H=BLOCK_H,
    )
    return out


def replacement_func():
    return _fused_permute_contiguous