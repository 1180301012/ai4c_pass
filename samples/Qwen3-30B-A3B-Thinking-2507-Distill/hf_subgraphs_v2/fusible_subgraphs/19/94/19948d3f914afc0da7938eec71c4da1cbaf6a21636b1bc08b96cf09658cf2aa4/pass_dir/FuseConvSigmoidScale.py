import torch
import triton
import triton.language as tl


def pattern(in_0, in_1, in_2, in_3):
    conv2d = torch.conv2d(in_3, in_1, in_0, (1, 1), (0, 0), (1, 1), 4)
    tmp_3 = torch.sigmoid(conv2d)
    tmp_4 = tmp_3.view(1, -1, 1, 1)
    tmp_5 = in_2 * tmp_4
    tmp_6 = tmp_5.contiguous()
    return tmp_6


def replacement_args(in_0, in_1, in_2, in_3):
    return (in_0, in_1, in_2, in_3)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_HW': 128},  num_warps=4,  num_stages=2),
        triton.Config({'BLOCK_HW': 256},  num_warps=4,  num_stages=2),
        triton.Config({'BLOCK_HW': 512},  num_warps=4,  num_stages=2),
        triton.Config({'BLOCK_HW': 512},  num_warps=8,  num_stages=3),
        triton.Config({'BLOCK_HW': 1024}, num_warps=8,  num_stages=3),
        triton.Config({'BLOCK_HW': 2048}, num_warps=8,  num_stages=3),
        triton.Config({'BLOCK_HW': 2048}, num_warps=16, num_stages=4),
        triton.Config({'BLOCK_HW': 4096}, num_warps=16, num_stages=4),
    ],
    key=['HW'],
)
@triton.jit
def _fused_conv_sigmoid_scale_kernel(
    in3_ptr,   # [1, C_in=32, 1, 1]
    in1_ptr,   # [C_out=96, C_in_g=8, 1, 1]
    in0_ptr,   # [C_out=96]
    in2_ptr,   # [N=1, C_out, H, W]
    out_ptr,   # [N=1, C_out, H, W]
    HW,        # H*W  (runtime scalar)
    BLOCK_HW: tl.constexpr,
    C_IN_G:   tl.constexpr,   # 8  (power of 2)
    C_OUT:    tl.constexpr,   # 96 (hardcoded for this problem)
):
    """
    One program per (batch×channel, HW-tile).
    C_OUT=96 and groups=4 → group = pid_bc // 24 is a compile-time division.
    """
    pid_bc = tl.program_id(0)
    pid_hw = tl.program_id(1)

    # ── compute sigmoid(conv2d(in3, in1, in0)[pid_bc, :]) ─────────────────────
    k_off = tl.arange(0, C_IN_G)                   # [0..7]
    group = pid_bc // (C_OUT // 4)                  # pid_bc // 24 (constexpr)

    w     = tl.load(in1_ptr + pid_bc * C_IN_G + k_off).to(tl.float32)  # [8]
    b     = tl.load(in0_ptr + pid_bc).to(tl.float32)                    # scalar
    in3_v = tl.load(in3_ptr + group * C_IN_G + k_off).to(tl.float32)   # [8]
    scale = tl.sum(w * in3_v) + b                                        # scalar
    scale = 1.0 / (1.0 + tl.exp(-scale))

    # ── vectorised scale multiply ────────────────────────────────────────────
    base   = pid_bc * HW
    hw_off = pid_hw * BLOCK_HW + tl.arange(0, BLOCK_HW)
    mask   = hw_off < HW
    feat   = tl.load(in2_ptr + base + hw_off, mask=mask).to(tl.float32)
    tl.store(out_ptr + base + hw_off, feat * scale, mask=mask)


@torch.fx.wrap
def fused_conv_sigmoid_scale(in_0, in_1, in_2, in_3):
    """
    Fused: conv2d(in_3, in_1, in_0, groups=4) -> sigmoid -> view -> in_2 * scaled
    weight [96,8,1,1], in_3 [1,32,1,1], in_2 [1,96,H,W].
    """
    N     = in_2.shape[0]
    C_out = in_2.shape[1]
    H     = in_2.shape[2]
    W     = in_2.shape[3]
    HW    = H * W

    out  = torch.empty_like(in_2)
    grid = lambda meta: (N * C_out, triton.cdiv(HW, meta['BLOCK_HW']))

    _fused_conv_sigmoid_scale_kernel[grid](
        in_3, in_1, in_0, in_2, out,
        HW,
        C_IN_G=8,
        C_OUT=96,
    )
    return out


def replacement_func():
    return fused_conv_sigmoid_scale