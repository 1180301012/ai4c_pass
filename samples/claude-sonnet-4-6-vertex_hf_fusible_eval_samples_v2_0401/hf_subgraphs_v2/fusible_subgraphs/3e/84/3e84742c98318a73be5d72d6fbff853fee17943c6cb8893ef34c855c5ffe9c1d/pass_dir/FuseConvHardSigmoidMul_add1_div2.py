import torch
import triton
import triton.language as tl


# Match post-conv ops only: hard_sigmoid(+1,/2) + broadcast_mul
# (conv2d excluded so the kernel only handles the elementwise chain,
#  guaranteeing bit-exact results since we don't touch the GEMM)
def pattern(conv_out, in_2):
    tmp_3 = conv_out + 1.0
    tmp_4 = tmp_3 / 2.0
    tmp_5 = tmp_4.clamp_(0.0, 1.0)
    tmp_6 = in_2 * tmp_5
    return (tmp_6,)


def replacement_args(conv_out, in_2):
    return (conv_out, in_2)


@triton.jit
def _hardsig_broadcast_mul_add1(
    conv_ptr,   # [BC]       – conv output (flattened [B, C, 1, 1])
    in2_ptr,    # [BC*HW]    – feature map (contiguous NCHW)
    out_ptr,    # [BC*HW]
    HW,
    BLOCK: tl.constexpr,
):
    # 1-D grid: one program per (batch, channel).
    # Inner loop over spatial tiles with SW-pipelining (num_stages).
    bc = tl.program_id(0)

    # Compute hard-sigmoid in fp32
    v  = tl.load(conv_ptr + bc).to(tl.float32)
    hs = (v + 1.0) * 0.5
    hs = tl.minimum(1.0, tl.maximum(0.0, hs))

    base = bc * HW
    for start in range(0, HW, BLOCK):
        offs = start + tl.arange(0, BLOCK)
        mask = offs < HW
        x    = tl.load(in2_ptr + base + offs, mask=mask, other=0.0)
        tl.store(out_ptr + base + offs, x * hs.to(x.dtype), mask=mask)


@torch.fx.wrap
def fused_hardsig_broadcast_mul_add1(conv_out, in_2):
    """
    conv_out : [B, C, 1, 1]   (conv2d output, already exact)
    in_2     : [B, C, H, W]
    """
    B  = in_2.shape[0]
    C  = in_2.shape[1]
    HW = in_2.shape[2] * in_2.shape[3]
    BC = B * C

    out = torch.empty_like(in_2)

    # Choose config by HW range (no autotune overhead)
    # Key: use BLOCK small enough to give ≥3 loop iterations so SW-pipelining
    # (num_stages) can fully overlap memory-latency with compute.
    if HW <= 64:
        BLOCK, nw, ns = 64,   2, 2   # 1 iter (HW fits in 1 BLOCK)
    elif HW <= 256:
        BLOCK, nw, ns = 64,   2, 4   # 3-4 iters; HW=192→3 full iters
    elif HW <= 512:
        BLOCK, nw, ns = 128,  4, 4   # 3-4 iters
    elif HW <= 2048:
        BLOCK, nw, ns = 256,  4, 4   # 5-8 iters for HW=512-2048
    else:
        BLOCK, nw, ns = 512,  8, 4   # many iters for large HW

    _hardsig_broadcast_mul_add1[(BC,)](
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
    return fused_hardsig_broadcast_mul_add1