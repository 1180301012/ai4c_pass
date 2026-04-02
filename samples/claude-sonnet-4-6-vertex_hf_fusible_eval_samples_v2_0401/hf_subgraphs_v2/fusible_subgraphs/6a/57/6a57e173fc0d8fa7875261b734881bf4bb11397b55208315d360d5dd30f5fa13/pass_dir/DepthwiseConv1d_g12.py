import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Depthwise 1-D convolution, kernel=65, same-padding=32, groups=12
# Input/Output shape: [B, G, S, C]  (G == groups == 12)
# Weight shape:       [G, 1, 65, 1]  →  flat [G * 65]
# ---------------------------------------------------------------------------
@triton.autotune(
    configs=[
        triton.Config({'TILE_S': 1,  'BLOCK_C':  64}),
        triton.Config({'TILE_S': 2,  'BLOCK_C':  64}),
        triton.Config({'TILE_S': 4,  'BLOCK_C':  64}),
        triton.Config({'TILE_S': 8,  'BLOCK_C':  64}),
        triton.Config({'TILE_S': 1,  'BLOCK_C': 128}),
        triton.Config({'TILE_S': 2,  'BLOCK_C': 128}),
        triton.Config({'TILE_S': 4,  'BLOCK_C': 128}),
    ],
    key=['B', 'S', 'C', 'OUTPUT_DTYPE'],
)
@triton.jit
def dw_conv1d_g12_kernel(
    x_ptr, w_ptr, out_ptr,
    B, G, S, C,
    OUTPUT_DTYPE: tl.constexpr,   # 0=f32, 1=f16, 2=bf16
    TILE_S: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    S_TILES = tl.cdiv(S, TILE_S)

    pid    = tl.program_id(0)
    g_idx  = pid % G
    tmp    = pid // G
    s_tile = tmp % S_TILES
    b_idx  = tmp // S_TILES

    s_base     = s_tile * TILE_S
    s_offs     = tl.arange(0, TILE_S)
    c_offs     = tl.arange(0, BLOCK_C)
    c_mask     = c_offs < C
    s_out_mask = (s_base + s_offs) < S

    bgs_stride = G * S * C
    gs_stride  = S * C

    acc = tl.zeros([TILE_S, BLOCK_C], dtype=tl.float32)

    for k in range(65):
        s_in_offs = s_base + s_offs + k - 32
        valid_s   = (s_in_offs >= 0) & (s_in_offs < S)

        w = tl.load(w_ptr + g_idx * 65 + k).to(tl.float32)

        in_off = (b_idx * bgs_stride + g_idx * gs_stride
                  + s_in_offs[:, None] * C + c_offs[None, :])
        vals = tl.load(x_ptr + in_off,
                       mask=valid_s[:, None] & c_mask[None, :],
                       other=0.0).to(tl.float32)
        acc += w * vals

    out_off = (b_idx * bgs_stride + g_idx * gs_stride
               + (s_base + s_offs[:, None]) * C + c_offs[None, :])
    out_mask = s_out_mask[:, None] & c_mask[None, :]

    if OUTPUT_DTYPE == 1:
        tl.store(out_ptr + out_off, acc.to(tl.float16),  mask=out_mask)
    elif OUTPUT_DTYPE == 2:
        tl.store(out_ptr + out_off, acc.to(tl.bfloat16), mask=out_mask)
    else:
        tl.store(out_ptr + out_off, acc.to(tl.float32),  mask=out_mask)


_DTYPE_MAP_G12 = {torch.float32: 0, torch.float16: 1, torch.bfloat16: 2}


@torch.fx.wrap
def triton_dw_conv1d_g12(in_0, in_2):
    """Replacement for torch.conv2d(..., groups=12) with 65×1 kernel."""
    B, G, S, C = in_2.shape
    output     = torch.empty_like(in_2)
    dtype_code = _DTYPE_MAP_G12.get(in_2.dtype, 0)

    grid = lambda meta: (B * G * triton.cdiv(S, meta['TILE_S']),)
    dw_conv1d_g12_kernel[grid](
        in_2, in_0, output,
        B, G, S, C,
        OUTPUT_DTYPE=dtype_code,
    )
    return output


def pattern(in_0, in_2):
    result = torch.conv2d(in_2, in_0, None, (1, 1), (32, 0), (1, 1), 12)
    return result


def replacement_args(in_0, in_2):
    return (in_0, in_2)


def replacement_func():
    return triton_dw_conv1d_g12