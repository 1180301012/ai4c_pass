import torch
import triton
import triton.language as tl


def pattern(conv_out, in_2):
    tmp_3 = conv_out + 3.0
    tmp_4 = tmp_3 / 6.0
    tmp_5 = tmp_4.clamp_(0.0, 1.0)
    tmp_6 = in_2 * tmp_5
    return (tmp_6,)


def replacement_args(conv_out, in_2):
    return (conv_out, in_2)


@triton.jit
def _hardsig_broadcast_mul_add3(
    conv_ptr,
    in2_ptr,
    out_ptr,
    HW,
    BLOCK: tl.constexpr,
):
    bc = tl.program_id(0)

    v  = tl.load(conv_ptr + bc).to(tl.float32)
    hs = (v + 3.0) * (1.0 / 6.0)
    hs = tl.minimum(1.0, tl.maximum(0.0, hs))

    base = bc * HW
    for start in range(0, HW, BLOCK):
        offs = start + tl.arange(0, BLOCK)
        mask = offs < HW
        x    = tl.load(in2_ptr + base + offs, mask=mask, other=0.0)
        tl.store(out_ptr + base + offs, x * hs.to(x.dtype), mask=mask)


@torch.fx.wrap
def fused_hardsig_broadcast_mul_add3(conv_out, in_2):
    B  = in_2.shape[0]
    C  = in_2.shape[1]
    HW = in_2.shape[2] * in_2.shape[3]
    BC = B * C

    out = torch.empty_like(in_2)

    if HW <= 64:
        BLOCK, nw, ns = 64,   2, 2
    elif HW <= 256:
        BLOCK, nw, ns = 64,   2, 4
    elif HW <= 512:
        BLOCK, nw, ns = 128,  4, 4
    elif HW <= 2048:
        BLOCK, nw, ns = 256,  4, 4
    else:
        BLOCK, nw, ns = 512,  8, 4

    _hardsig_broadcast_mul_add3[(BC,)](
        conv_out.reshape(BC),
        in_2.contiguous().reshape(-1),
        out.reshape(-1),
        HW,
        BLOCK=BLOCK,
        num_warps=nw,
        num_stages=ns,
    )

    return out


def replacement_func():
    return fused_hardsig_broadcast_mul_add3