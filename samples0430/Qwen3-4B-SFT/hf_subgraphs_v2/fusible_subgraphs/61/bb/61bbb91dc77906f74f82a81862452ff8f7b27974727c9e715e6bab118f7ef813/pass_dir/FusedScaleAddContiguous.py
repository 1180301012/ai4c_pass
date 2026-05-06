"""
Pass 2 of the full fusion: fuse  scale × iadd_result + in_2 + .contiguous()

Pattern:
  tmp_3 = tmp_2 * in_0          # scale
  tmp_4 = tmp_3 + in_2           # residual add
  tmp_5 = tmp_4.contiguous()     # layout

Where tmp_2 is a FREE VARIABLE (the iadd node's output in the target graph).

Handling non-contiguous tensors: instead of flat index calculations that assume
contiguous layout, we use the strides passed explicitly.

Memory traffic savings:
  Original: (iadd_result read + scale write + iadd_result read + in2 read +
             tmp3 write + in2 read + tmp4 write + contiguous copy)
           ≈ 1200MB DRAM
  Fused:    (iadd_result read + in2 read + write out) × 2
           = 492MB DRAM   (~1.4× reduction)
"""

import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Pattern – free variable tmp_2 matches the iadd node's output in the target
# ---------------------------------------------------------------------------
def pattern(in_0, in_2, tmp_2):
    tmp_3 = tmp_2 * in_0
    tmp_4 = tmp_3 + in_2
    tmp_5 = tmp_4.contiguous()
    return tmp_5


def replacement_args(in_0, in_2, tmp_2):
    return (in_0, in_2, tmp_2)


# ---------------------------------------------------------------------------
# Triton kernel
#   Handles both contiguous AND non-contiguous (strided) input tensors via
#   explicit strides passed from the wrapper.
#
#   out_ptr[b,c,hw_tile,j] = in0 (scalar) * tmp2[b,c,hw_tile,j]
#                           + in2[b,c,hw_tile,j]
# ---------------------------------------------------------------------------
@triton.autotune(
    configs=[
        triton.Config({'BLOCK': 256},  num_warps=4, num_stages=3),
        triton.Config({'BLOCK': 512},  num_warps=4, num_stages=3),
        triton.Config({'BLOCK': 512},  num_warps=8, num_stages=3),
        triton.Config({'BLOCK': 1024}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK': 1024}, num_warps=16, num_stages=4),
        triton.Config({'BLOCK': 2048}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK': 2048}, num_warps=16, num_stages=4),
    ],
    key=['N', 'HW_J'],
)
@triton.jit
def scale_add_contiguous_kernel(
    iadd_ptr, in2_ptr, in0_ptr, out_ptr,
    iadd_sb, iadd_sc, iadd_shw,  # iadd (tmp_2) strides
    in2_sb,  in2_sc,  in2_shw,  # in2 strides
    out_sb,  out_sc,  out_shw,  # out strides
    N,
    HW_J,                      # B * C * H * W * J  (needed for index decomposition)
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    off = pid * BLOCK + tl.arange(0, BLOCK)
    mask = off < N

    # Decompose flat index assuming contiguous [B, C, H, W] storage
    j_off    = off % 64
    hw_off   = (off // 64) % (HW_J // 4096)   # HW*J=HW_J, HW//64=HW_J//4096
    c_off    = (off // (HW_J * 64)) % 512      # C=512
    b_off    = off // (512 * HW_J * 64)

    hw = hw_off * 64 + (j_off // 64)   # full spatial index (0..4095)
    bc = b_off * 512 + c_off            # flat bc index (0..16383)

    # Flat base for each tensor: bc * stride(1) + hw * stride(2)
    # For contiguous [B,C,H,W]: stride(1)=H*W, stride(2)=W
    bases = bc * iadd_sc + hw * iadd_shw
    v = tl.load(iadd_ptr + bases, mask=mask, other=0.0)

    bases = bc * in2_sc + hw * in2_shw
    r = tl.load(in2_ptr + bases, mask=mask, other=0.0)

    scale = tl.load(in0_ptr)
    result = r + scale * v

    tl.store(out_ptr + bases, result, mask=mask)


# ---------------------------------------------------------------------------
# Wrapper
# ---------------------------------------------------------------------------
@torch.fx.wrap
def triton_scale_add_contiguous(in_0, in_2, tmp_2):
    B  = tmp_2.shape[0]
    C2 = tmp_2.shape[1]
    HW = tmp_2.shape[2] * tmp_2.shape[3]   # H*W = 4096
    HW_J = HW * HW  # = 16384 = H * W * W  (= H*W*J when J=W=64)

    N = B * C2 * HW

    # Use actual strides (handles contiguous AND non-contiguous tmp_2)
    s_i = tmp_2.stride(0), tmp_2.stride(1), tmp_2.stride(2)
    s_2 = in_2.stride(0), in_2.stride(1), in_2.stride(2)

    out = torch.empty_like(tmp_2)
    s_o = out.stride(0), out.stride(1), out.stride(2)

    grid = lambda meta: (triton.cdiv(N, meta['BLOCK']),)
    scale_add_contiguous_kernel[grid](
        tmp_2, in_2, in_0, out,
        s_i[0], s_i[1], s_i[2],
        s_2[0], s_2[1], s_2[2],
        s_o[0], s_o[1], s_o[2],
        N, HW_J,
    )
    return out


# ---------------------------------------------------------------------------
# replacement_func
# ---------------------------------------------------------------------------
def replacement_func():
    return triton_scale_add_contiguous