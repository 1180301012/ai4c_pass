import torch
import torch.fx
import triton
import triton.language as tl
import inspect


# ─────────────────────────────────────────────────────────────────────────────
# Pattern: manually constructed graph so force_args_symbolic_trace is bypassed.
# gm.graph from torch.compile keeps F.interpolate as a leaf with original
# kwargs, but force_args_symbolic_trace would flatten them to positional args
# causing a silent length mismatch.  By providing a torch.fx.Graph directly
# (isinstance check in _replace_pattern), we hand the exact node format.
# ─────────────────────────────────────────────────────────────────────────────

class _InterpolatePatternGraph(torch.fx.Graph):
    """Graph subclass with a __call__ signature so inspect.signature works."""
    def __call__(self, tensor):          # only for inspect.signature introspection
        pass                             # never actually called


def _build_interpolate_pattern():
    g = _InterpolatePatternGraph()
    tensor = g.placeholder('tensor')
    result = g.call_function(
        torch.nn.functional.interpolate,
        args=(tensor,),
        kwargs={'size': (128, 128), 'mode': 'bilinear', 'align_corners': False},
    )
    g.output((result,))
    return g


pattern = _build_interpolate_pattern()


def replacement_args(tensor):
    return (tensor,)


# ─────────────────────────────────────────────────────────────────────────────
# Triton kernel: bilinear upsampling from any NCHW → (N, C, 128, 128)
# ─────────────────────────────────────────────────────────────────────────────

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_S': 128}, num_warps=4),
        triton.Config({'BLOCK_S': 256}, num_warps=4),
        triton.Config({'BLOCK_S': 256}, num_warps=8),
        triton.Config({'BLOCK_S': 512}, num_warps=8),
        triton.Config({'BLOCK_S': 1024}, num_warps=8),
        triton.Config({'BLOCK_S': 128}, num_warps=8),
        triton.Config({'BLOCK_S': 512}, num_warps=4),
        triton.Config({'BLOCK_S': 1024}, num_warps=16),
    ],
    key=['N', 'C', 'IH', 'IW', 'OH', 'OW'],
)
@triton.jit
def bilinear_upsample_nchw_kernel(
    input_ptr,
    output_ptr,
    N, C, IH, IW, OH, OW,
    sN, sC, sH, sW,
    BLOCK_S: tl.constexpr,
):
    """
    Grid: (N*C, cdiv(OH*OW, BLOCK_S))
    Supports arbitrary strides so .contiguous() is not needed.
    Output is always contiguous NCHW.
    """
    nc      = tl.program_id(0)
    s_block = tl.program_id(1)

    n = nc // C
    c = nc % C

    s_start   = s_block * BLOCK_S
    s_offsets = s_start + tl.arange(0, BLOCK_S)
    s_mask    = s_offsets < (OH * OW)

    oh = s_offsets // OW
    ow = s_offsets % OW

    # align_corners=False: src = (dst + 0.5) * (src/dst) - 0.5
    scale_h = IH.to(tl.float32) / OH.to(tl.float32)
    scale_w = IW.to(tl.float32) / OW.to(tl.float32)

    ih_f = (oh.to(tl.float32) + 0.5) * scale_h - 0.5
    iw_f = (ow.to(tl.float32) + 0.5) * scale_w - 0.5

    ih_floor = tl.floor(ih_f)
    iw_floor = tl.floor(iw_f)

    ih0 = tl.maximum(0, tl.minimum(IH - 1, ih_floor.to(tl.int32)))
    ih1 = tl.maximum(0, tl.minimum(IH - 1, (ih_floor + 1.0).to(tl.int32)))
    iw0 = tl.maximum(0, tl.minimum(IW - 1, iw_floor.to(tl.int32)))
    iw1 = tl.maximum(0, tl.minimum(IW - 1, (iw_floor + 1.0).to(tl.int32)))

    wh = ih_f - ih_floor
    ww = iw_f - iw_floor

    # Base pointer for this (n, c) slice using arbitrary strides
    in_base = n * sN + c * sC

    v00 = tl.load(input_ptr + in_base + ih0 * sH + iw0 * sW, mask=s_mask, other=0.0)
    v01 = tl.load(input_ptr + in_base + ih0 * sH + iw1 * sW, mask=s_mask, other=0.0)
    v10 = tl.load(input_ptr + in_base + ih1 * sH + iw0 * sW, mask=s_mask, other=0.0)
    v11 = tl.load(input_ptr + in_base + ih1 * sH + iw1 * sW, mask=s_mask, other=0.0)

    # Compute in float32 for accuracy, cast back to original dtype
    orig_dtype = v00.dtype
    v00f = v00.to(tl.float32)
    v01f = v01.to(tl.float32)
    v10f = v10.to(tl.float32)
    v11f = v11.to(tl.float32)

    result_f = (1.0 - wh) * (1.0 - ww) * v00f + \
               (1.0 - wh) * ww          * v01f + \
               wh          * (1.0 - ww) * v10f + \
               wh          * ww          * v11f

    result = result_f.to(orig_dtype)

    # Output is always contiguous NCHW
    out_base = nc * (OH * OW)
    tl.store(output_ptr + out_base + oh * OW + ow, result, mask=s_mask)


@torch.fx.wrap
def triton_bilinear_upsample_128(tensor):
    """
    Triton bilinear upsampling to (128, 128). Handles non-contiguous inputs.
    Input:  [N, C, IH, IW]
    Output: [N, C, 128, 128]
    """
    N, C, IH, IW = tensor.shape
    sN, sC, sH, sW = tensor.stride()
    OH, OW = 128, 128
    NC = N * C

    output = torch.empty((N, C, OH, OW), dtype=tensor.dtype, device=tensor.device)

    def grid(meta):
        return (NC, triton.cdiv(OH * OW, meta['BLOCK_S']))

    bilinear_upsample_nchw_kernel[grid](
        tensor, output,
        N, C, IH, IW, OH, OW,
        sN, sC, sH, sW,
    )

    return output



def replacement_func():
    return triton_bilinear_upsample_128