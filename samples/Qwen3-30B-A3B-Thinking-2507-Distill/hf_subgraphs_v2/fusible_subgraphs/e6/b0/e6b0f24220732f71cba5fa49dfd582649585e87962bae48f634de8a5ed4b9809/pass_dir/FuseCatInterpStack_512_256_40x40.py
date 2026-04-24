import torch
import triton
import triton.language as tl


# ── Match only torch.cat((in_2, in_3), 1) ───────────────────────────────────
# This is the simplest pattern that avoids the interpolate tracing issue and
# definitively tells us whether torch.cat is matchable.
def pattern(in_2, in_3):
    return torch.cat((in_2, in_3), 1)


def replacement_args(in_2, in_3):
    return (in_2, in_3)


# ── Triton kernel: flatten-copy src into the right channel slice of out ───────
@triton.autotune(
    configs=[
        triton.Config({'BLOCK': 1024}, num_warps=4),
        triton.Config({'BLOCK': 2048}, num_warps=8),
        triton.Config({'BLOCK': 4096}, num_warps=8),
    ],
    key=['N'],   # autotune once per unique N — reuse config across different batch sizes sharing the same N
)
@triton.jit
def _flat_copy(src_ptr, dst_ptr, N, B, C_src, C_dst, HW, C_offset, BLOCK: tl.constexpr):
    """
    Copy N = B*C_src*HW elements from src into the C_offset channel slice of dst.
    dst[b, C_offset+c, hw] = src[b, c, hw]
    dst-batch-stride = C_dst*HW,  src-batch-stride = C_src*HW
    """
    pid  = tl.program_id(0)
    off  = pid * BLOCK + tl.arange(0, BLOCK)
    mask = off < N
    vals = tl.load(src_ptr + off, mask=mask)
    b    = off // (C_src * HW)
    c_hw = off %  (C_src * HW)
    d_off = b * C_dst * HW + C_offset * HW + c_hw
    tl.store(dst_ptr + d_off, vals, mask=mask)


@torch.fx.wrap
def triton_cat2(in_2, in_3):
    B, C_src, H, W = in_2.shape
    HW    = H * W
    C_dst = 2 * C_src
    N     = B * C_src * HW
    out   = torch.empty((B, C_dst, H, W), dtype=in_2.dtype, device=in_2.device)
    grid  = lambda m: (triton.cdiv(N, m['BLOCK']),)
    # in_2 → out[:, 0 : C_src, :, :]
    _flat_copy[grid](in_2, out, N, B, C_src, C_dst, HW, C_offset=0)
    # in_3 → out[:, C_src : C_dst, :, :]
    _flat_copy[grid](in_3, out, N, B, C_src, C_dst, HW, C_offset=C_src)
    return out


def replacement_func():
    return triton_cat2