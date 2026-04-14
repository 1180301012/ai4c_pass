import torch
import triton
import triton.language as tl


# ── Pattern ──────────────────────────────────────────────────────────────────
def pattern(in_0, in_1, in_2, in_3):
    conv2d = torch.conv2d(in_3, in_1, in_0, (1, 1), (0, 0), (1, 1), 4)
    tmp_3 = torch.sigmoid(conv2d)
    tmp_4 = tmp_3.view(1, -1, 1, 1)
    tmp_5 = in_2 * tmp_4
    tmp_6 = tmp_5.contiguous()
    return (tmp_6,)


# ── Argument extraction ───────────────────────────────────────────────────────
def replacement_args(in_0, in_1, in_2, in_3):
    return (in_0, in_1, in_2, in_3)


# ── Triton kernel ─────────────────────────────────────────────────────────────
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_HW': 128},  num_warps=2),
        triton.Config({'BLOCK_HW': 256},  num_warps=4),
        triton.Config({'BLOCK_HW': 512},  num_warps=4),
        triton.Config({'BLOCK_HW': 1024}, num_warps=4),
        triton.Config({'BLOCK_HW': 2048}, num_warps=8),
        triton.Config({'BLOCK_HW': 4096}, num_warps=8),
    ],
    key=['HW'],
)
@triton.jit
def _fused_conv_sigmoid_mul_kernel(
    # pointers
    in3_ptr,     # [C_in]        conv input  (flat)
    weight_ptr,  # [C_out, C_in_G]  conv weight (flat 2-D)
    bias_ptr,    # [C_out]       conv bias
    in2_ptr,     # [C_out, HW]   feature map (flat 2-D)
    out_ptr,     # [C_out, HW]   output      (flat 2-D)
    # scalars
    C,           # number of output channels (96)
    HW,          # H * W  (varies per graph)
    # constexprs – enables loop unrolling and compile-time arith
    G:       tl.constexpr,       # groups = 4
    C_out_G: tl.constexpr,       # output channels per group = 24
    C_in_G:  tl.constexpr,       # input  channels per group = 8
    BLOCK_HW: tl.constexpr,
):
    # 2-D grid: axis-0 = output channel, axis-1 = HW tile
    c      = tl.program_id(0)   # [0, C)
    pid_hw = tl.program_id(1)

    # ── grouped 1×1 conv for channel c ───────────────────────────────────────
    g        = c // C_out_G               # which group [0, G)
    base_in  = g * C_in_G                 # first input channel of this group
    base_w   = c * C_in_G                 # start of weights for this output channel

    conv_val = tl.load(bias_ptr + c).to(tl.float32)

    for k in range(C_in_G):              # C_in_G is constexpr → unrolled
        w = tl.load(weight_ptr + base_w + k).to(tl.float32)
        x = tl.load(in3_ptr   + base_in + k).to(tl.float32)
        conv_val += w * x

    # sigmoid
    sig_val = 1.0 / (1.0 + tl.exp(-conv_val))

    # ── broadcast-multiply with in_2[c, :] ───────────────────────────────────
    hw_base    = pid_hw * BLOCK_HW
    hw_offsets = hw_base + tl.arange(0, BLOCK_HW)
    hw_mask    = hw_offsets < HW

    ch_offset  = c * HW
    in2_vals   = tl.load(in2_ptr + ch_offset + hw_offsets, mask=hw_mask)
    out_vals   = in2_vals * sig_val.to(in2_vals.dtype)
    tl.store(out_ptr + ch_offset + hw_offsets, out_vals, mask=hw_mask)


# ── Kernel wrapper ────────────────────────────────────────────────────────────
@torch.fx.wrap
def fused_conv_sigmoid_mul(in_0, in_1, in_2, in_3):
    """
    in_0 : bias    [C_out]               (C_out = 96)
    in_1 : weight  [C_out, C_in_G, 1, 1] ([96, 8, 1, 1])
    in_2 : feature [1, C_out, H, W]
    in_3 : input   [1, C_in,  1, 1]      ([1, 32, 1, 1])

    All tensors are contiguous, so pointer arithmetic in the kernel maps:
      in_3[0, in_c, 0, 0]  ->  in3_ptr  + in_c        (stride-1 dim is channels)
      in_1[c, k, 0, 0]     ->  weight_ptr + c*C_in_G+k
      in_0[c]              ->  bias_ptr + c
      in_2[0, c, h, w]     ->  in2_ptr  + c*HW + hw
    No reshape/view needed – pass the raw tensors directly.
    """
    B, C, H, W_dim = in_2.shape          # (1, 96, H, W)
    HW      = H * W_dim
    G       = 4
    C_out_G = C // G                      # 24
    C_in_G  = in_3.shape[1] // G         # 8

    out = torch.empty_like(in_2)

    grid = lambda meta: (C, triton.cdiv(HW, meta['BLOCK_HW']))

    _fused_conv_sigmoid_mul_kernel[grid](
        in_3, in_1, in_0, in_2, out,     # raw tensors – no reshape
        C, HW,
        G, C_out_G, C_in_G,
    )

    return (out,)


# ── Replacement function (zero-argument, returns callable) ────────────────────
def replacement_func():
    return fused_conv_sigmoid_mul