import torch
import triton
import triton.language as tl


# ─── Pattern: match the ENTIRE model.py forward ─────────────────────────────
# in_0: bias  [96]
# in_1: weight [96, 8, 1, 1]   (groups=4, IC/group=8, OC/group=24)
# in_2: feature map [1, 96, H, W]
# in_3: input  [1, 32, 1, 1]
def pattern(in_0, in_1, in_2, in_3):
    conv2d = torch.conv2d(in_3, in_1, in_0, (1, 1), (0, 0), (1, 1), 4)
    tmp_3 = torch.sigmoid(conv2d)
    tmp_4 = tmp_3.view(1, -1, 1, 1)
    tmp_5 = in_2 * tmp_4
    tmp_6 = tmp_5.contiguous()
    return tmp_6


def replacement_args(in_0, in_1, in_2, in_3):
    return (in_0, in_1, in_2, in_3)


# ─── Single fused Triton kernel ──────────────────────────────────────────────
# Grid: (C=96 channels, ceil(N / BLOCK_SIZE) spatial blocks)
#
# Each program (pid_c, pid_s):
#   1. Computes grouped-conv dot product for channel pid_c (8 FMAs in float32).
#   2. Applies sigmoid to get per-channel scale.
#   3. Casts scale to native dtype and multiplies spatial block of x.
#
# Constexprs:
#   IC_PER_GROUP = 32 // 4 = 8
#   OC_PER_GROUP = 96 // 4 = 24

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 256}, num_warps=2),
    ],
    key=['N'],
)
@triton.jit
def _fused_gconv_sigmoid_scale_kernel(
    bias_ptr,    # [C]           – in_0
    weight_ptr,  # [C * IC_PG]   – in_1 flattened (1×1 kernel)
    x_ptr,       # [1, C, H, W]  – in_2
    inp_ptr,     # [G * IC_PG]   – in_3 flattened
    out_ptr,     # [1, C, H, W]  – output
    N,           # H * W
    IC_PER_GROUP: tl.constexpr,   # 8
    OC_PER_GROUP: tl.constexpr,   # 24
    BLOCK_SIZE:   tl.constexpr,
):
    pid_c = tl.program_id(0)   # output channel 0..95
    pid_s = tl.program_id(1)   # spatial block

    # ── Step 1: grouped conv for this channel (float32 accumulation) ─────────
    g           = pid_c // OC_PER_GROUP
    inp_base    = g * IC_PER_GROUP
    weight_base = pid_c * IC_PER_GROUP

    acc = tl.load(bias_ptr + pid_c).to(tl.float32)
    for i in range(IC_PER_GROUP):
        inp_val = tl.load(inp_ptr    + inp_base    + i).to(tl.float32)
        w_val   = tl.load(weight_ptr + weight_base + i).to(tl.float32)
        acc    += inp_val * w_val

    # ── Step 2: sigmoid scale ────────────────────────────────────────────────
    scale = tl.sigmoid(acc)   # float32

    # ── Step 3: load spatial block, scale in float32, cast back, store ───────
    offsets = pid_s * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask    = offsets < N
    base    = pid_c * N

    x   = tl.load(x_ptr + base + offsets, mask=mask, other=0.0)
    out = (x.to(tl.float32) * scale).to(x.dtype)
    tl.store(out_ptr + base + offsets, out, mask=mask)


# ─── FX-wrapped launcher ─────────────────────────────────────────────────────
@torch.fx.wrap
def fused_gconv_sigmoid_scale_wrapper(in_0, in_1, in_2, in_3):
    """
    Single-kernel replacement for:
        conv2d(in_3, in_1, in_0, stride=1, pad=0, dilation=1, groups=4)
        → sigmoid → view(1,-1,1,1) → in_2 * scale → contiguous()
    """
    C  = in_2.shape[1]                    # 96
    N  = in_2.shape[2] * in_2.shape[3]   # H * W

    IC_PER_GROUP = 8    # 32 in_channels // 4 groups
    OC_PER_GROUP = 24   # 96 out_channels // 4 groups

    out = torch.empty_like(in_2)

    _fused_gconv_sigmoid_scale_kernel[
        (C, triton.cdiv(N, 256))
    ](
        bias_ptr=in_0,
        weight_ptr=in_1,
        x_ptr=in_2,
        inp_ptr=in_3,
        out_ptr=out,
        N=N,
        IC_PER_GROUP=IC_PER_GROUP,
        OC_PER_GROUP=OC_PER_GROUP,
    )

    return out


def replacement_func():
    return fused_gconv_sigmoid_scale_wrapper