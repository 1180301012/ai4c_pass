import torch
import triton
import triton.language as tl


@triton.jit
def _stage1_depthwise_conv_add_store_stats_kernel(
    x_ptr,
    conv_w_ptr,
    conv_b_ptr,
    tmp7_ptr,
    part_sum_ptr,
    part_sumsq_ptr,
    CHANNELS: tl.constexpr,
    NCBLOCKS: tl.constexpr,
    MAX_CBLOCKS: tl.constexpr,
    BLOCK_T: tl.constexpr,
    BLOCK_C: tl.constexpr,
    IS_BF16: tl.constexpr,
):
    pid_t = tl.program_id(0)
    pid_c = tl.program_id(1)

    offs_t = pid_t * BLOCK_T + tl.arange(0, BLOCK_T)
    offs_c = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)

    mask_t = offs_t < 256
    mask_c = offs_c < CHANNELS
    mask_tc = mask_t[:, None] & mask_c[None, :]

    h = offs_t // 16
    w = offs_t % 16

    bias = tl.load(conv_b_ptr + offs_c, mask=mask_c, other=0.0).to(tl.float32)
    acc = tl.zeros((BLOCK_T, BLOCK_C), dtype=tl.float32) + bias[None, :]

    center_idx = offs_c[None, :] * 256 + offs_t[:, None]
    center = tl.load(x_ptr + center_idx, mask=mask_tc, other=0.0).to(tl.float32)

    for kh in range(3):
        ih = h + kh - 1
        valid_h = (ih >= 0) & (ih < 16)
        for kw in range(3):
            iw = w + kw - 1
            valid_hw = valid_h & (iw >= 0) & (iw < 16)
            idx_x = offs_c[None, :] * 256 + (ih[:, None] * 16 + iw[:, None])
            x_vals = tl.load(x_ptr + idx_x, mask=mask_c[None, :] & valid_hw[:, None], other=0.0).to(tl.float32)
            w_vals = tl.load(conv_w_ptr + offs_c * 9 + kh * 3 + kw, mask=mask_c, other=0.0).to(tl.float32)
            acc += x_vals * w_vals[None, :]

    acc += center

    if IS_BF16:
        stored = acc.to(tl.bfloat16)
        stat_vals = stored.to(tl.float32)
    else:
        stored = acc
        stat_vals = acc

    out_idx = offs_t[:, None] * CHANNELS + offs_c[None, :]
    tl.store(tmp7_ptr + out_idx, stored, mask=mask_tc)

    partial_sum = tl.sum(stat_vals, axis=1)
    partial_sumsq = tl.sum(stat_vals * stat_vals, axis=1)
    part_idx = offs_t * MAX_CBLOCKS + pid_c
    tl.store(part_sum_ptr + part_idx, partial_sum, mask=mask_t)
    tl.store(part_sumsq_ptr + part_idx, partial_sumsq, mask=mask_t)


@triton.jit
def _stage2_layernorm_transpose_kernel(
    tmp7_ptr,
    ln_w_ptr,
    ln_b_ptr,
    part_sum_ptr,
    part_sumsq_ptr,
    out_ptr,
    CHANNELS: tl.constexpr,
    NCBLOCKS: tl.constexpr,
    MAX_CBLOCKS: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    token = tl.program_id(0)
    pid_c = tl.program_id(1)

    red_offs = tl.arange(0, MAX_CBLOCKS)
    red_mask = red_offs < NCBLOCKS
    sums = tl.load(part_sum_ptr + token * MAX_CBLOCKS + red_offs, mask=red_mask, other=0.0)
    sumsqs = tl.load(part_sumsq_ptr + token * MAX_CBLOCKS + red_offs, mask=red_mask, other=0.0)

    sum_all = tl.sum(sums, axis=0)
    sumsq_all = tl.sum(sumsqs, axis=0)
    mean = sum_all / CHANNELS
    var = sumsq_all / CHANNELS - mean * mean
    inv_std = tl.rsqrt(var + 1e-5)

    offs_c = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)
    mask_c = offs_c < CHANNELS
    idx = token * CHANNELS + offs_c

    vals = tl.load(tmp7_ptr + idx, mask=mask_c, other=0.0).to(tl.float32)
    gamma = tl.load(ln_w_ptr + offs_c, mask=mask_c, other=1.0).to(tl.float32)
    beta = tl.load(ln_b_ptr + offs_c, mask=mask_c, other=0.0).to(tl.float32)
    y = (vals - mean) * inv_std
    y = y * gamma + beta
    tl.store(out_ptr + idx, y, mask=mask_c)


@torch.fx.wrap
def fused_depthwise_conv_add_flatten_transpose_layernorm_transpose_dispatch(
    ln_b,
    ln_w,
    conv_b,
    conv_w,
    x,
    route,
):
    if route == "c768":
        channels = 768
        is_bf16 = True
    elif route == "c1024":
        channels = 1024
        is_bf16 = False
    else:
        raise RuntimeError("unsupported route")

    max_cblocks = 8
    n_cblocks = triton.cdiv(channels, 128)

    tmp7 = torch.empty((1, 256, channels), device=x.device, dtype=x.dtype)
    out = torch.empty((256, 1, channels), device=x.device, dtype=x.dtype)
    part_sum = torch.empty((256, max_cblocks), device=x.device, dtype=torch.float32)
    part_sumsq = torch.empty((256, max_cblocks), device=x.device, dtype=torch.float32)

    grid1 = (triton.cdiv(256, 8), n_cblocks)
    _stage1_depthwise_conv_add_store_stats_kernel[grid1](
        x,
        conv_w,
        conv_b,
        tmp7,
        part_sum,
        part_sumsq,
        CHANNELS=channels,
        NCBLOCKS=n_cblocks,
        MAX_CBLOCKS=max_cblocks,
        BLOCK_T=8,
        BLOCK_C=128,
        IS_BF16=is_bf16,
        num_warps=4,
        num_stages=2,
    )

    grid2 = (256, n_cblocks)
    _stage2_layernorm_transpose_kernel[grid2](
        tmp7,
        ln_w,
        ln_b,
        part_sum,
        part_sumsq,
        out,
        CHANNELS=channels,
        NCBLOCKS=n_cblocks,
        MAX_CBLOCKS=max_cblocks,
        BLOCK_C=128,
        num_warps=4,
        num_stages=2,
    )

    return (tmp7, out, out)


def shared_replacement_func():
    return fused_depthwise_conv_add_flatten_transpose_layernorm_transpose_dispatch