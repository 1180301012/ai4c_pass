import torch
import triton
import triton.language as tl


# Match the ENTIRE computation: conv2d + sigmoid + mul + gelu + pool + flatten + dropout
def pattern(in_0, in_1, in_2, in_3):
    conv2d = torch.conv2d(in_3, in_1, in_0, (1, 1), (0, 0), (1, 1), 1)
    tmp_3 = conv2d.sigmoid()
    tmp_4 = in_2 * tmp_3
    tmp_5 = torch.nn.functional.gelu(tmp_4, approximate='none')
    tmp_6 = torch.nn.functional.adaptive_avg_pool2d(tmp_5, 1)
    tmp_7 = tmp_6.flatten(1, -1)
    tmp_8 = torch.nn.functional.dropout(tmp_7, 0.0, False, False)
    return tmp_8


def replacement_args(in_0, in_1, in_2, in_3):
    # in_0 = bias  [C]
    # in_1 = weight [C, K, 1, 1]
    # in_2 = feature map [B, C, H, W]
    # in_3 = conv input [B, K, 1, 1]
    return (in_0, in_1, in_2, in_3)


# ── Full-fusion kernel: GEMM (conv1x1) + sigmoid + mul + GELU + avgpool ──────
# Grid: (B * C,) — one program per (batch, output-channel) pair.
# BLOCK_K=64 is fixed (always equals the input channel count for this problem).
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_HW': 64},  num_warps=2, num_stages=2),
        triton.Config({'BLOCK_HW': 64},  num_warps=4, num_stages=2),
        triton.Config({'BLOCK_HW': 128}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_HW': 128}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_HW': 256}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_HW': 256}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_HW': 64},  num_warps=2, num_stages=3),
        triton.Config({'BLOCK_HW': 128}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_HW': 256}, num_warps=8, num_stages=3),
    ],
    key=['B', 'HW', 'DTYPE_SIZE'],
)
@triton.jit
def fused_conv_sigmoid_mul_gelu_pool_kernel(
    bias_ptr,       # [C]
    weight_ptr,     # [C, K]
    in2_ptr,        # [B, C, HW]
    conv_in_ptr,    # [B, K]
    out_ptr,        # [B, C]
    B, C, K, HW, DTYPE_SIZE,   # added for dtype-specific autotune
    BLOCK_HW: tl.constexpr,   # spatial elements per iteration (autotuned)
    BLOCK_K:  tl.constexpr,   # == K (64), compile-time constant
):
    bc  = tl.program_id(0)
    b   = bc // C
    c   = bc % C

    # 1×1 conv = dot-product over K
    conv_val = tl.load(bias_ptr + c).to(tl.float32)
    for k in tl.static_range(BLOCK_K):
        x_k = tl.load(conv_in_ptr + b * K + k).to(tl.float32)
        w_k = tl.load(weight_ptr  + c * K + k).to(tl.float32)
        conv_val = conv_val + x_k * w_k
    sig_val = 1.0 / (1.0 + tl.exp(-conv_val))

    # GELU + avgpool
    base = bc * HW
    acc  = 0.0
    for hw_start in range(0, HW, BLOCK_HW):
        offs = hw_start + tl.arange(0, BLOCK_HW)
        mask = offs < HW
        x = tl.load(in2_ptr + base + offs, mask=mask, other=0.0).to(tl.float32)
        x = x * sig_val
        gelu = 0.5 * x * (1.0 + tl.math.erf(x * 0.7071067811865476))
        acc  = acc + tl.sum(gelu, axis=0)

    tl.store(out_ptr + bc, acc / tl.cast(HW, tl.float32))


@torch.fx.wrap
def fused_conv_sigmoid_mul_gelu_avgpool(bias, weight, in2, conv_in):
    B, C, H, W = in2.shape
    K  = conv_in.shape[1]
    HW = H * W
    BC = B * C

    out = torch.empty((B, C), dtype=in2.dtype, device=in2.device)

    fused_conv_sigmoid_mul_gelu_pool_kernel[(BC,)](
        bias, weight, in2, conv_in, out,
        B=B, C=C, K=K, HW=HW,
        DTYPE_SIZE=in2.element_size(),   # dtype-specific autotune key
        BLOCK_K=K,   # constexpr = 64 for all our graphs
    )
    return out


def replacement_func():
    return fused_conv_sigmoid_mul_gelu_avgpool