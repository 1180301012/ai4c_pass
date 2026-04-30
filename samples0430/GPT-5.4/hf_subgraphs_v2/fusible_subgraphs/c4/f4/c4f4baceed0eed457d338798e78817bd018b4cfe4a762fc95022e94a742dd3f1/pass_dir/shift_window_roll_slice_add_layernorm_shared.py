import torch
import triton
import triton.language as tl


@triton.jit
def _shift_window_add_layernorm_kernel(
    bias_ptr,
    weight_ptr,
    residual_ptr,
    src_ptr,
    out_add_ptr,
    out_ln_ptr,
    rows,
    bias_s0,
    weight_s0,
    residual_s0,
    residual_s1,
    residual_s2,
    src_s0,
    src_s1,
    src_s2,
    src_s3,
    src_s4,
    src_s5,
    out_add_s0,
    out_add_s1,
    out_add_s2,
    out_ln_s0,
    out_ln_s1,
    out_ln_s2,
    EPS: tl.constexpr,
    H_TOTAL: tl.constexpr,
    CROP: tl.constexpr,
    C: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    row = tl.program_id(0)
    if row >= rows:
        return

    tokens_per_batch = CROP * CROP
    batch = row // tokens_per_batch
    token = row % tokens_per_batch
    h = token // CROP
    w = token % CROP

    src_h = tl.where(h >= 3, h - 3, h + H_TOTAL - 3)
    src_w = tl.where(w >= 3, w - 3, w + H_TOTAL - 3)

    gh = src_h // 7
    ih = src_h % 7
    gw = src_w // 7
    iw = src_w % 7

    cols = tl.arange(0, BLOCK_C)
    mask = cols < C

    residual_offsets = batch * residual_s0 + token * residual_s1 + cols * residual_s2
    src_offsets = (
        batch * src_s0
        + gh * src_s1
        + ih * src_s2
        + gw * src_s3
        + iw * src_s4
        + cols * src_s5
    )
    out_add_offsets = batch * out_add_s0 + token * out_add_s1 + cols * out_add_s2
    out_ln_offsets = batch * out_ln_s0 + token * out_ln_s1 + cols * out_ln_s2

    residual = tl.load(residual_ptr + residual_offsets, mask=mask, other=0)
    shifted = tl.load(src_ptr + src_offsets, mask=mask, other=0)
    added = residual + shifted
    tl.store(out_add_ptr + out_add_offsets, added, mask=mask)

    added_fp32 = added.to(tl.float32)
    mean = tl.sum(added_fp32, axis=0) / C
    centered = added_fp32 - mean
    var = tl.sum(centered * centered, axis=0) / C
    inv_std = tl.rsqrt(var + EPS)

    weight = tl.load(weight_ptr + cols * weight_s0, mask=mask, other=1).to(tl.float32)
    bias = tl.load(bias_ptr + cols * bias_s0, mask=mask, other=0).to(tl.float32)
    out_ln = centered * inv_std * weight + bias
    tl.store(out_ln_ptr + out_ln_offsets, out_ln, mask=mask)


def _launch_shift_window_add_layernorm(bias, weight, residual, src, out_add, out_ln, h_total, crop, channels, block_c, num_warps):
    rows = residual.size(0) * residual.size(1)
    grid = (rows,)
    _shift_window_add_layernorm_kernel[grid](
        bias,
        weight,
        residual,
        src,
        out_add,
        out_ln,
        rows,
        bias.stride(0),
        weight.stride(0),
        residual.stride(0),
        residual.stride(1),
        residual.stride(2),
        src.stride(0),
        src.stride(1),
        src.stride(2),
        src.stride(3),
        src.stride(4),
        src.stride(5),
        out_add.stride(0),
        out_add.stride(1),
        out_add.stride(2),
        out_ln.stride(0),
        out_ln.stride(1),
        out_ln.stride(2),
        EPS=1e-5,
        H_TOTAL=h_total,
        CROP=crop,
        C=channels,
        BLOCK_C=block_c,
        num_warps=num_warps,
        num_stages=2,
    )


@torch.fx.wrap
def replacement_dispatch(bias, weight, residual, src, route):
    out_add = torch.empty_like(residual)
    out_ln = torch.empty_like(residual)

    if route == "crop128_c96":
        _launch_shift_window_add_layernorm(
            bias,
            weight,
            residual,
            src,
            out_add,
            out_ln,
            h_total=133,
            crop=128,
            channels=96,
            block_c=128,
            num_warps=4,
        )
    elif route == "crop64_c192":
        _launch_shift_window_add_layernorm(
            bias,
            weight,
            residual,
            src,
            out_add,
            out_ln,
            h_total=70,
            crop=64,
            channels=192,
            block_c=256,
            num_warps=4,
        )
    elif route == "crop32_c384":
        _launch_shift_window_add_layernorm(
            bias,
            weight,
            residual,
            src,
            out_add,
            out_ln,
            h_total=35,
            crop=32,
            channels=384,
            block_c=512,
            num_warps=8,
        )
    else:
        raise RuntimeError(f"Unknown route: {route}")

    return out_add, out_ln