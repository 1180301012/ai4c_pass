import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Pattern: sigmoid(in_2) -> view(1,-1,1,1) -> expand_as(in_1) -> in_1 * scale
# Fuses 4 ops into 1 Triton kernel, saving 3 kernel launch overheads.
# ---------------------------------------------------------------------------

def pattern(in_1, in_2):
    tmp_0 = in_2.sigmoid()
    tmp_1 = tmp_0.view(1, -1, 1, 1)
    tmp_2 = tmp_1.expand_as(in_1)
    tmp_3 = in_1 * tmp_2
    return tmp_3


def replacement_args(in_1, in_2):
    return (in_1, in_2)


# ---------------------------------------------------------------------------
# Triton kernel: sigmoid-scale broadcast multiply.
# HW is tl.constexpr → inner loop unrolled at compile time.
# Autotune picks best (BLOCK_HW, num_warps) per (C, HW) key.
# ---------------------------------------------------------------------------

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_HW': 64},  num_warps=1),
        triton.Config({'BLOCK_HW': 64},  num_warps=2),
        triton.Config({'BLOCK_HW': 64},  num_warps=4),
        triton.Config({'BLOCK_HW': 128}, num_warps=4),
        triton.Config({'BLOCK_HW': 128}, num_warps=8),
        triton.Config({'BLOCK_HW': 256}, num_warps=4),
        triton.Config({'BLOCK_HW': 256}, num_warps=8),
        triton.Config({'BLOCK_HW': 256}, num_warps=16),
    ],
    key=['C', 'HW'],
)
@triton.jit
def fused_sigmoid_scale_mul_kernel(
    in1_ptr,          # [B, C, H, W]  contiguous NCHW
    in2_ptr,          # [B, 1, C]     → in2[0, 0, c] = in2_ptr[c]
    out_ptr,          # [B, C, H, W]
    C,
    HW:       tl.constexpr,
    BLOCK_HW: tl.constexpr,
):
    c      = tl.program_id(0)
    scale  = tl.sigmoid(tl.load(in2_ptr + c).to(tl.float32))
    base   = c * HW
    for start in range(0, HW, BLOCK_HW):
        offs = start + tl.arange(0, BLOCK_HW)
        mask = offs < HW
        val  = tl.load(in1_ptr + base + offs, mask=mask, other=0.0).to(tl.float32)
        tl.store(out_ptr + base + offs, val * scale, mask=mask)


@torch.fx.wrap
def triton_sigmoid_scale_mul(in_1, in_2):
    """Fused: sigmoid(in_2).view(1,-1,1,1).expand_as(in_1) * in_1"""
    B, C, H, W = in_1.shape
    HW  = H * W
    out = torch.empty_like(in_1)
    fused_sigmoid_scale_mul_kernel[(C,)](in_1, in_2, out, C, HW)
    return out


def replacement_func():
    return triton_sigmoid_scale_mul