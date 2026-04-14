"""
Fused pass: conv2d(1x1, 1x1 spatial input) + hardsigmoid + mul + avgpool + flatten + dropout

Single fused kernel:
  1. Tiled GEMM for the 1×1 convolution (tensor cores via tl.dot)
  2. hardsigmoid on the GEMM result
  3. Weighted mean of in_2 (the SE-attention scale applied to the feature map)
  Output = hardsigmoid(conv(x_se)) * mean(x)
"""
import torch
import triton
import triton.language as tl


# ─────────────────────────────────────────────────────────────────────────────
# Pattern & replacement_args
# ─────────────────────────────────────────────────────────────────────────────

def pattern(bias, weight, x, x_se):
    """
    bias   = in_0  [C]
    weight = in_1  [C, C, 1, 1]
    x      = in_2  [B, C, H, W]   (feature map)
    x_se   = in_3  [B, C, 1, 1]   (SE attention input)
    """
    conv_out = torch.conv2d(x_se, weight, bias, (1, 1), (0, 0), (1, 1), 1)
    hs  = torch.nn.functional.hardsigmoid(conv_out, False)
    m   = x * hs
    p   = torch.nn.functional.adaptive_avg_pool2d(m, 1)
    f   = p.flatten(1, -1)
    d   = torch.nn.functional.dropout(f, 0.0, False, False)
    return d


def replacement_args(bias, weight, x, x_se):
    return (bias, weight, x, x_se)


# ─────────────────────────────────────────────────────────────────────────────
# Single Fused Kernel:
#   GEMM  →  hardsigmoid  →  weighted mean of x  →  output
# ─────────────────────────────────────────────────────────────────────────────
@triton.autotune(
    configs=[
        # (B=1,  C=1024, HW=144)
        triton.Config({'BLOCK_B': 16, 'BLOCK_C_OUT': 32, 'BLOCK_C_IN': 32, 'BLOCK_HW': 16}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_B': 16, 'BLOCK_C_OUT': 64, 'BLOCK_C_IN': 32, 'BLOCK_HW': 16}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_B': 16, 'BLOCK_C_OUT': 64, 'BLOCK_C_IN': 64, 'BLOCK_HW': 16}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_B': 16, 'BLOCK_C_OUT': 64, 'BLOCK_C_IN': 32, 'BLOCK_HW': 32}, num_warps=4, num_stages=3),
        # (B=32, C=1024, HW=64)
        triton.Config({'BLOCK_B': 32, 'BLOCK_C_OUT': 32, 'BLOCK_C_IN': 32, 'BLOCK_HW': 16}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_B': 32, 'BLOCK_C_OUT': 64, 'BLOCK_C_IN': 32, 'BLOCK_HW': 16}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_B': 32, 'BLOCK_C_OUT': 64, 'BLOCK_C_IN': 64, 'BLOCK_HW': 32}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_B': 32, 'BLOCK_C_OUT': 64, 'BLOCK_C_IN': 64, 'BLOCK_HW': 64}, num_warps=8, num_stages=3),
        # (B=128, C=1024, HW=64 or HW=144)
        triton.Config({'BLOCK_B': 64, 'BLOCK_C_OUT': 32, 'BLOCK_C_IN': 32, 'BLOCK_HW': 16}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_B': 64, 'BLOCK_C_OUT': 64, 'BLOCK_C_IN': 32, 'BLOCK_HW': 16}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_B': 64, 'BLOCK_C_OUT': 64, 'BLOCK_C_IN': 64, 'BLOCK_HW': 16}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_B': 64, 'BLOCK_C_OUT': 64, 'BLOCK_C_IN': 32, 'BLOCK_HW': 32}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_B': 64, 'BLOCK_C_OUT': 64, 'BLOCK_C_IN': 64, 'BLOCK_HW': 32}, num_warps=8, num_stages=3),
    ],
    key=['B', 'C_in', 'C_out', 'HW'],
)
@triton.jit
def _fused_conv1x1_hardsigmoid_avgpool_kernel(
    inp_ptr,    # x_se  [B, C_in]   (contiguous; same as [B, C, 1, 1])
    wt_ptr,     # weight [C_out, C_in]
    bias_ptr,   # bias  [C_out]
    y_ptr,      # in_2  [B*C_out, HW]
    out_ptr,    # output [B, C_out]
    B, C_in, C_out, HW,
    BLOCK_B:     tl.constexpr,
    BLOCK_C_OUT: tl.constexpr,
    BLOCK_C_IN:  tl.constexpr,
    BLOCK_HW:    tl.constexpr,
):
    # Map 1-D pid → (B-tile, C_out-tile)
    num_c_tiles = (C_out + BLOCK_C_OUT - 1) // BLOCK_C_OUT
    pid    = tl.program_id(0)
    pid_b  = pid // num_c_tiles
    pid_c  = pid  % num_c_tiles

    b_start = pid_b * BLOCK_B
    c_start = pid_c * BLOCK_C_OUT

    b_offs  = b_start + tl.arange(0, BLOCK_B)       # [BLOCK_B]
    c_offs  = c_start + tl.arange(0, BLOCK_C_OUT)   # [BLOCK_C_OUT]
    b_mask  = b_offs < B
    c_mask  = c_offs < C_out

    # ── GEMM  [BLOCK_B, C_in] @ [C_in, BLOCK_C_OUT]  →  acc ───────────────
    acc = tl.zeros([BLOCK_B, BLOCK_C_OUT], dtype=tl.float32)

    for k in range(0, C_in, BLOCK_C_IN):
        k_offs = k + tl.arange(0, BLOCK_C_IN)
        k_mask = k_offs < C_in

        a = tl.load(inp_ptr + b_offs[:, None] * C_in + k_offs[None, :],
                    mask=b_mask[:, None] & k_mask[None, :], other=0.0).to(tl.float32)
        b_mat = tl.load(wt_ptr + c_offs[:, None] * C_in + k_offs[None, :],
                        mask=c_mask[:, None] & k_mask[None, :], other=0.0).to(tl.float32)
        acc += tl.dot(a, tl.trans(b_mat))

    # ── bias + hardsigmoid ──────────────────────────────────────────────────
    bias = tl.load(bias_ptr + c_offs, mask=c_mask, other=0.0).to(tl.float32)
    conv_val = acc + bias[None, :]
    hs = tl.minimum(tl.maximum((conv_val + 3.0) * (1.0 / 6.0), 0.0), 1.0)

    # ── weighted mean of y[b, c_out, :] ────────────────────────────────────
    # y layout: y[bc, hw] = y_ptr + bc * HW + hw,  bc = b * C_out + c_out
    bc_base = b_offs[:, None] * C_out + c_offs[None, :]  # [BLOCK_B, BLOCK_C_OUT]
    bc_mask2 = b_mask[:, None] & c_mask[None, :]

    y_sum = tl.zeros([BLOCK_B, BLOCK_C_OUT], dtype=tl.float32)

    for hw_start in range(0, HW, BLOCK_HW):
        hw_offs = hw_start + tl.arange(0, BLOCK_HW)   # [BLOCK_HW]
        hw_mask = hw_offs < HW

        # 3-D offsets: [BLOCK_B, BLOCK_C_OUT, BLOCK_HW]
        y_offs = bc_base[:, :, None] * HW + hw_offs[None, None, :]
        y_mask = bc_mask2[:, :, None] & hw_mask[None, None, :]

        y_tile = tl.load(y_ptr + y_offs, mask=y_mask, other=0.0).to(tl.float32)
        y_sum  += tl.sum(y_tile, axis=2)           # [BLOCK_B, BLOCK_C_OUT]

    out_vals = hs * y_sum / HW

    # ── store ────────────────────────────────────────────────────────────────
    tl.store(out_ptr + bc_base, out_vals, mask=bc_mask2)


# ─────────────────────────────────────────────────────────────────────────────
# Replacement function
# ─────────────────────────────────────────────────────────────────────────────
@torch.fx.wrap
def fused_conv1x1_hardsigmoid_avgpool(bias, weight, x, x_se):
    """
    bias   [C]            in_0
    weight [C, C, 1, 1]   in_1
    x      [B, C, H, W]   in_2  (feature map)
    x_se   [B, C, 1, 1]   in_3  (SE input)
    returns [B, C]
    """
    B     = x.shape[0]
    C_out = x.shape[1]
    H     = x.shape[2]
    W     = x.shape[3]
    HW    = H * W
    C_in  = C_out   # 1×1 conv preserves channel count

    out = torch.empty(B, C_out, dtype=x.dtype, device=x.device)

    _fused_conv1x1_hardsigmoid_avgpool_kernel[
        lambda meta: (
            ((B + meta['BLOCK_B'] - 1) // meta['BLOCK_B']) *
            ((C_out + meta['BLOCK_C_OUT'] - 1) // meta['BLOCK_C_OUT']),
        )
    ](
        x_se, weight, bias, x, out,
        B, C_in, C_out, HW,
    )

    return out


def replacement_func():
    return fused_conv1x1_hardsigmoid_avgpool