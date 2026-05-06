import torch
import triton
import triton.language as tl


# ──────────────────────────────────────────────────────────────────────────────
# Pattern: x.permute(2,0,1).reshape(-1,3,384,384)
#
# Matches BOTH instances in the model graph:
#   (a) unfold(in_1, stride=192).permute(2,0,1).reshape → tmp_2
#   (b) unfold(in_2, stride=288).permute(2,0,1).reshape → tmp_5
#
# x for (a) has shape [1,3,147456,4]  contiguous (stride_g=4)
# x for (b) has shape [1,3,147456,25] contiguous (stride_g=2)
# Output: contiguous [4,3,384,384] or [25,3,384,384]
# ──────────────────────────────────────────────────────────────────────────────
def pattern(x):
    return x.permute(2, 0, 1).reshape(-1, 3, 384, 384)


def replacement_args(x):
    return (x,)


# ── Triton gather kernel ──────────────────────────────────────────────
# Reads from src[n, c, r, g] and writes to dst[g, c, r].
# src strides handled via constexpr strides_n, strides_g.
# ──────────────────────────────────────────────────────────────────────
@triton.autotune(
    configs=[
        triton.Config({'BLOCK': 512},  num_warps=4),
        triton.Config({'BLOCK': 1024}, num_warps=4),
        triton.Config({'BLOCK': 2048}, num_warps=4),
        triton.Config({'BLOCK': 4096}, num_warps=8),
    ],
    key=[],
)
@triton.jit
def _pyr_kernel(
    src_ptr, dst_ptr,
    N,      # = C * 384 * 384 = 147456
    G,      # number of patches (P)
    S,      # kernel size = stride_out (384)
    stride_n,   # stride of dim-1 (C-stride in source)
    stride_g,   # stride of last dim in source (2 or 4 in this model)
    BLOCK: tl.constexpr,
):
    pid  = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    total = G * N
    mask  = offs < total

    g       = offs // N          # patch index  (0 .. G-1)
    bc_r    = offs % N           # position in C*S*S
    c       = bc_r // (S * S)
    r       = bc_r  % (S * S)
    kh      = r // S
    kw      = r  % S

    # Element source position: src[c, g, kh*S + kw]
    # Batch dim always 0, so offset = c*stride_n + g*stride_g + kh*S + kw
    src_off = c * stride_n + g * stride_g + kh * S + kw

    x = tl.load(src_ptr + src_off, mask=mask, other=0.0)
    tl.store(dst_ptr + offs, x, mask=mask)


@torch.fx.wrap
def permute_reshape_fn(x):
    """
    x : unfold output [1, 3, N, G_patches, W_out] or [1, 3, N, G_patches]
    returns contiguous [G_patches, 3, 384, 384]
    Uses strides to handle both 4D and 5D unfold outputs.
    """
    C      = 3
    S      = 384
    N      = x.stride(0) // (x.shape[1] * x.stride(1))   # = H*W = C*S*S = 147456
    G      = x.stride(-2)   # stride of the P dimension (足够 for 5-D unfold outputs)
    G      = G if G > 1 else x.stride(-1)  # fall back to W dimension stride
    out    = torch.empty((G, C, S, S), dtype=x.dtype, device=x.device)
    total  = G * N
    grid   = lambda m: (triton.cdiv(total, m['BLOCK']),)
    _pyr_kernel[grid](x, out, N, G, S, x.stride(1), x.stride(-1))
    return out


def replacement_func():
    return permute_reshape_fn