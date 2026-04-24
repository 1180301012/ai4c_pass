import torch
import triton
import triton.language as tl


def pattern(relu_output, in_0):
    tmp_3 = torch.nn.functional.avg_pool2d(relu_output, 3, 1, 1, False, False, None)
    tmp_4 = tmp_3 - relu_output
    tmp_5 = in_0.unsqueeze(-1)
    tmp_6 = tmp_5.unsqueeze(-1)
    tmp_7 = tmp_6 * tmp_4
    tmp_8 = relu_output + tmp_7
    return tmp_8


def replacement_args(relu_output, in_0):
    return (relu_output, in_0)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 256},  num_warps=4),
        triton.Config({'BLOCK_SIZE': 512},  num_warps=4),
        triton.Config({'BLOCK_SIZE': 512},  num_warps=8),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=16),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=16),
    ],
    key=['N_total'],
)
@triton.jit
def fused_relu_avgpool_scale_kernel(
    scale_ptr,   # [C]
    feat_ptr,    # [B, C, H, W]
    output_ptr,  # [B, C, H, W]
    N_total,
    C, H, W,
    IS_FP16: tl.constexpr,
    IS_BF16: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < N_total

    # Decompose flat offset into (b, c, h, w)
    w = offs % W
    tmp = offs // W
    h = tmp % H
    tmp = tmp // H
    c = tmp % C
    b = tmp // C

    # Channel scale (1-d vector, broadcast across W)
    scale = tl.load(scale_ptr + c, mask=mask, other=0.0).to(tl.float32)

    # Base pointer for this (b, c, h, :)
    base = (b * C + c) * H * W + h * W

    # 3x3-neighborhood load with element-wise boundary masking.
    # All masks are derived from the 1-d tensors w/h → no Python if/else.
    x_c  = tl.load(feat_ptr + base + w,         mask=mask, other=0.0).to(tl.float32)
    is_l = w > 0
    is_r = w < (W - 1)
    is_t = h > 0
    is_b = h < (H - 1)

    base_m1 = base - W
    base_p1 = base + W

    x_tl = tl.load(feat_ptr + base_m1 + w - 1, mask=mask & is_t & is_l, other=0.0).to(tl.float32)
    x_tc = tl.load(feat_ptr + base_m1 + w,     mask=mask & is_t,        other=0.0).to(tl.float32)
    x_tr = tl.load(feat_ptr + base_m1 + w + 1, mask=mask & is_t & is_r, other=0.0).to(tl.float32)

    x_ml = tl.load(feat_ptr + base + w - 1, mask=mask & is_l, other=0.0).to(tl.float32)
    x_mr = tl.load(feat_ptr + base + w + 1, mask=mask & is_r, other=0.0).to(tl.float32)

    x_bl = tl.load(feat_ptr + base_p1 + w - 1, mask=mask & is_b & is_l, other=0.0).to(tl.float32)
    x_bc = tl.load(feat_ptr + base_p1 + w,     mask=mask & is_b,        other=0.0).to(tl.float32)
    x_br = tl.load(feat_ptr + base_p1 + w + 1, mask=mask & is_b & is_r, other=0.0).to(tl.float32)

    avg_val = (x_tl + x_tc + x_tr + x_ml + x_c + x_mr + x_bl + x_bc + x_br) * (1.0 / 9.0)

    # out = x_c * (1 + scale*(avg_val/9 - 1)) = x_c*(scale*avg_val/9 + (1-scale))
    out = x_c * (scale * avg_val * (1.0 / 9.0) + (1.0 - scale))

    if IS_FP16:
        out_conv = out.to(tl.float16)
    elif IS_BF16:
        out_conv = out.to(tl.bfloat16)
    else:
        out_conv = out

    tl.store(output_ptr + base + w, out_conv, mask=mask)


@torch.fx.wrap
def fused_relu_avgpool_scale(relu_output, in_0):
    B = relu_output.shape[0]
    C = relu_output.shape[1]
    H = relu_output.shape[2]
    W = relu_output.shape[3]
    N_total = B * C * H * W

    output = torch.empty_like(relu_output)

    IS_FP16 = relu_output.dtype == torch.float16
    IS_BF16 = relu_output.dtype == torch.bfloat16

    grid = lambda meta: ((N_total + meta['BLOCK_SIZE'] - 1) // meta['BLOCK_SIZE'],)

    fused_relu_avgpool_scale_kernel[grid](
        in_0, relu_output, output,
        N_total,
        C, H, W,
        IS_FP16=IS_FP16,
        IS_BF16=IS_BF16,
    )

    return output


def replacement_func():
    return fused_relu_avgpool_scale