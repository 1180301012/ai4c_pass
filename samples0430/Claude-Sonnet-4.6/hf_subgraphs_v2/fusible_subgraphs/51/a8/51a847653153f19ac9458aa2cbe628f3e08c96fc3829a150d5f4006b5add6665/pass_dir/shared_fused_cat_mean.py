"""
Shared Triton kernel + dispatch wrapper for:
  cat([in_0, in_1], dim=1) → slice[:, :N, :, :] → mean((2,3), keepdim=True)

Architecture:
  _triton_fused_cat_mean  ← @torch.fx.wrap (opaque, runs actual Triton kernels)
  dispatch_fused_cat_mean ← plain function (FX-traceable), unpacks (out, mean)
                            so replace_pattern sees TWO outputs, matching the
                            pattern's (tmp_1, tmp_2).

All four pass files import `dispatch_fused_cat_mean` and return it from
`replacement_func()`, giving the framework exactly ONE unique replacement_func.
"""
import torch
import triton
import triton.language as tl


# ── Triton kernel ─────────────────────────────────────────────────────────────

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 64},   num_warps=2),
        triton.Config({'BLOCK_SIZE': 128},  num_warps=4),
        triton.Config({'BLOCK_SIZE': 256},  num_warps=4),
        triton.Config({'BLOCK_SIZE': 512},  num_warps=8),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=8),
    ],
    key=['HW'],
)
@triton.jit
def _fused_copy_mean_kernel(
    src_ptr, dst_ptr, mean_ptr,
    C, C2, HW,
    src_bs, src_cs,
    dst_bs, dst_cs,
    c_offset,         # 0 for in_0 half, C for in_1 half
    BLOCK_SIZE: tl.constexpr,
):
    """
    Grid: (B * C,).  pid = b*C + c.
    Copies src[b,c,:,:] → dst[b, c_offset+c, :, :] and stores
    the spatial mean (float32 → auto-cast by Triton) to mean[b, c_offset+c, 0, 0].
    """
    pid   = tl.program_id(0)
    b     = pid // C
    c     = pid % C
    c_out = c + c_offset

    src_base  = src_ptr  + b * src_bs  + c     * src_cs
    dst_base  = dst_ptr  + b * dst_bs  + c_out * dst_cs

    acc = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for start in range(0, HW, BLOCK_SIZE):
        offs = start + tl.arange(0, BLOCK_SIZE)
        mask = offs < HW
        x    = tl.load(src_base + offs, mask=mask, other=0.0)
        tl.store(dst_base + offs, x, mask=mask)
        acc  = acc + x.to(tl.float32)

    total    = tl.sum(acc, axis=0)
    mean_val = total / HW
    # mean_ptr is contiguous [B, C2, 1, 1]: flat idx = b*C2 + c_out
    tl.store(mean_ptr + b * C2 + c_out, mean_val)


# ── Opaque inner function (@torch.fx.wrap) ────────────────────────────────────
# FX sees a single node `_triton_fused_cat_mean(in_0, in_1)` that returns a tuple.

@torch.fx.wrap
def _triton_fused_cat_mean(in_0, in_1):
    B  = in_0.shape[0]
    C  = in_0.shape[1]
    H  = in_0.shape[2]
    W  = in_0.shape[3]
    HW = H * W
    C2 = 2 * C

    out      = torch.empty([B, C2, H, W],     dtype=in_0.dtype, device=in_0.device)
    mean_out = torch.empty([B, C2, 1, 1],     dtype=in_0.dtype, device=in_0.device)

    grid = (B * C,)

    _fused_copy_mean_kernel[grid](
        in_0, out, mean_out,
        C, C2, HW,
        in_0.stride(0), in_0.stride(1),
        out.stride(0),  out.stride(1),
        0,
    )
    _fused_copy_mean_kernel[grid](
        in_1, out, mean_out,
        C, C2, HW,
        in_1.stride(0), in_1.stride(1),
        out.stride(0),  out.stride(1),
        C,
    )
    return (out, mean_out)


# ── FX-traceable outer wrapper ────────────────────────────────────────────────
# replace_pattern traces this function.  FX sees:
#   result  = call[_triton_fused_cat_mean](in_0, in_1)
#   output0 = getitem(result, 0)    ← maps to pattern output tmp_1
#   output1 = getitem(result, 1)    ← maps to pattern output tmp_2
# This gives the framework TWO distinct output nodes, matching the 2-tuple
# returned by the pattern function.

def dispatch_fused_cat_mean(in_0, in_1):
    result = _triton_fused_cat_mean(in_0, in_1)
    return result[0], result[1]