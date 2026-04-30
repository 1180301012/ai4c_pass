import torch
import triton
import triton.language as tl


def pattern(in_0, in_1, in_2, in_3):
    tmp_2 = torch.nn.functional.relu(in_2, inplace=False)
    tmp_3 = in_1 * tmp_2
    tmp_4 = tmp_3 + in_0
    tmp_5 = torch.nn.functional.max_pool2d(in_3, 2, 1, 0, 1, ceil_mode=True, return_indices=False)
    tmp_6 = torch.cat([tmp_5, tmp_4], dim=1)
    return tmp_6


def replacement_args(in_0, in_1, in_2, in_3):
    return (in_0, in_1, in_2, in_3)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_H': 8, 'BLOCK_W': 32, 'num_warps': 4}),
        triton.Config({'BLOCK_H': 8, 'BLOCK_W': 64, 'num_warps': 4}),
        triton.Config({'BLOCK_H': 8, 'BLOCK_W': 128, 'num_warps': 8}),
        triton.Config({'BLOCK_H': 16, 'BLOCK_W': 32, 'num_warps': 4}),
        triton.Config({'BLOCK_H': 16, 'BLOCK_W': 64, 'num_warps': 8}),
    ],
    key=['H', 'W', 'TOTAL_PLANES'],
)
@triton.jit
def fused_pool_pointwise_cat_kernel(
    bias_ptr,
    scale_ptr,
    x2_ptr,
    x3_ptr,
    out_ptr,
    N,
    C_PW,
    C_POOL,
    H,
    W,
    X2_SN,
    X2_SC,
    X2_SH,
    X2_SW,
    X3_SN,
    X3_SC,
    X3_SH,
    X3_SW,
    OUT_SN,
    OUT_SC,
    OUT_SH,
    OUT_SW,
    TOTAL_PLANES,
    BLOCK_H: tl.constexpr,
    BLOCK_W: tl.constexpr,
):
    pid_w = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_plane = tl.program_id(2)

    if pid_plane >= TOTAL_PLANES:
        return

    offs_w = pid_w * BLOCK_W + tl.arange(0, BLOCK_W)
    offs_h = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
    mask_hw = (offs_h[:, None] < H) & (offs_w[None, :] < W)

    total_c = C_POOL + C_PW
    n_idx = pid_plane // total_c
    c_out = pid_plane % total_c

    out_base = out_ptr + n_idx * OUT_SN + c_out * OUT_SC
    out_offsets = offs_h[:, None] * OUT_SH + offs_w[None, :] * OUT_SW

    if c_out < C_POOL:
        in_base = x3_ptr + n_idx * X3_SN + c_out * X3_SC
        top_left = tl.load(in_base + offs_h[:, None] * X3_SH + offs_w[None, :] * X3_SW, mask=mask_hw, other=-float('inf'))
        top_right = tl.load(in_base + offs_h[:, None] * X3_SH + (offs_w[None, :] + 1) * X3_SW, mask=mask_hw, other=-float('inf'))
        bot_left = tl.load(in_base + (offs_h[:, None] + 1) * X3_SH + offs_w[None, :] * X3_SW, mask=mask_hw, other=-float('inf'))
        bot_right = tl.load(in_base + (offs_h[:, None] + 1) * X3_SH + (offs_w[None, :] + 1) * X3_SW, mask=mask_hw, other=-float('inf'))
        val = tl.maximum(tl.maximum(top_left, top_right), tl.maximum(bot_left, bot_right))
    else:
        c_pw = c_out - C_POOL
        in_base = x2_ptr + n_idx * X2_SN + c_pw * X2_SC
        x = tl.load(in_base + offs_h[:, None] * X2_SH + offs_w[None, :] * X2_SW, mask=mask_hw, other=0.0)
        zero = tl.zeros([BLOCK_H, BLOCK_W], dtype=x.dtype)
        x = tl.maximum(x, zero)
        scale = tl.load(scale_ptr)
        bias = tl.load(bias_ptr)
        val = x * scale + bias

    tl.store(out_base + out_offsets, val, mask=mask_hw)


@torch.fx.wrap
def fused_pool_pointwise_cat(in_0, in_1, in_2, in_3):
    n = in_2.shape[0]
    c_pw = in_2.shape[1]
    h = in_2.shape[2]
    w = in_2.shape[3]
    c_pool = in_3.shape[1]
    out = torch.empty((n, c_pool + c_pw, h, w), device=in_2.device, dtype=in_2.dtype)

    total_planes = n * (c_pool + c_pw)
    grid = lambda meta: (
        triton.cdiv(w, meta['BLOCK_W']),
        triton.cdiv(h, meta['BLOCK_H']),
        total_planes,
    )

    fused_pool_pointwise_cat_kernel[grid](
        in_0,
        in_1,
        in_2,
        in_3,
        out,
        n,
        c_pw,
        c_pool,
        h,
        w,
        in_2.stride(0),
        in_2.stride(1),
        in_2.stride(2),
        in_2.stride(3),
        in_3.stride(0),
        in_3.stride(1),
        in_3.stride(2),
        in_3.stride(3),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        out.stride(3),
        total_planes,
    )
    return out


def replacement_func():
    return fused_pool_pointwise_cat