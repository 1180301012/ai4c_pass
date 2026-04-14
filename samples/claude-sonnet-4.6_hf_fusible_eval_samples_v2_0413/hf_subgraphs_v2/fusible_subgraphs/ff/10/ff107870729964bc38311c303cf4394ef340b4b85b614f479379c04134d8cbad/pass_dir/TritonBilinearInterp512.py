import torch
import triton
import triton.language as tl


def pattern(x):
    result = torch.nn.functional.interpolate(x, size=(512, 512), mode='bilinear', align_corners=False)
    return result


def replacement_args(x):
    return (x,)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 512},  num_warps=4),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=8),
    ],
    key=['N', 'C', 'IH', 'IW'],
)
@triton.jit
def bilinear_interp_512_kernel(
    input_ptr,
    output_ptr,
    N, C, IH, IW,
    IS_BF16: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Fixed output size: 512 x 512
    OH: tl.constexpr = 512
    OW: tl.constexpr = 512

    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    total = N * C * OH * OW
    mask = offs < total

    # Decode flat index → (n, c, oh, ow)
    ow_idx = offs % OW
    oh_idx = (offs // OW) % OH
    c_idx  = (offs // (OW * OH)) % C
    n_idx  = offs // (OW * OH * C)

    # ── Source coordinates (align_corners=False) ──
    #   src = (dst + 0.5) * (IN_SIZE / OUT_SIZE) - 0.5
    ih_f = IH.to(tl.float32)
    iw_f = IW.to(tl.float32)

    h_f = (oh_idx.to(tl.float32) + 0.5) * (ih_f / OH) - 0.5
    w_f = (ow_idx.to(tl.float32) + 0.5) * (iw_f / OW) - 0.5

    # ── Correct floor for negative values ──
    # Triton int32 cast truncates toward zero; subtract 1 when src < floor.
    h0_int = h_f.to(tl.int32)
    w0_int = w_f.to(tl.int32)
    h0_raw = h0_int + tl.where(h_f < h0_int.to(tl.float32), -1, 0)
    w0_raw = w0_int + tl.where(w_f < w0_int.to(tl.float32), -1, 0)
    h1_raw = h0_raw + 1
    w1_raw = w0_raw + 1

    # Fractional parts ∈ [0, 1)
    dh = h_f - h0_raw.to(tl.float32)
    dw = w_f - w0_raw.to(tl.float32)

    # Clamp to valid spatial range
    h0 = tl.maximum(tl.minimum(h0_raw, IH - 1), 0)
    h1 = tl.maximum(tl.minimum(h1_raw, IH - 1), 0)
    w0 = tl.maximum(tl.minimum(w0_raw, IW - 1), 0)
    w1 = tl.maximum(tl.minimum(w1_raw, IW - 1), 0)

    # Base offset for (n, c) slice in NCHW layout
    base = (n_idx * C + c_idx) * IH * IW

    # Load 4 corners, accumulate in float32
    v00 = tl.load(input_ptr + base + h0 * IW + w0, mask=mask, other=0.0).to(tl.float32)
    v01 = tl.load(input_ptr + base + h0 * IW + w1, mask=mask, other=0.0).to(tl.float32)
    v10 = tl.load(input_ptr + base + h1 * IW + w0, mask=mask, other=0.0).to(tl.float32)
    v11 = tl.load(input_ptr + base + h1 * IW + w1, mask=mask, other=0.0).to(tl.float32)

    # Bilinear blend
    result = (v00 * (1.0 - dh) * (1.0 - dw)
            + v01 * (1.0 - dh) * dw
            + v10 * dh * (1.0 - dw)
            + v11 * dh * dw)

    # Store with dtype matching input
    if IS_BF16:
        tl.store(output_ptr + offs, result.to(tl.bfloat16), mask=mask)
    else:
        tl.store(output_ptr + offs, result.to(tl.float16), mask=mask)


@torch.fx.wrap
def triton_bilinear_interp_512(x):
    N, C, IH, IW = x.shape
    OH, OW = 512, 512
    n_elements = N * C * OH * OW

    out = torch.empty((N, C, OH, OW), dtype=x.dtype, device=x.device)

    IS_BF16 = (x.dtype == torch.bfloat16)

    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)

    bilinear_interp_512_kernel[grid](
        x, out,
        N, C, IH, IW,
        IS_BF16=IS_BF16,
    )

    return out


def replacement_func():
    return triton_bilinear_interp_512