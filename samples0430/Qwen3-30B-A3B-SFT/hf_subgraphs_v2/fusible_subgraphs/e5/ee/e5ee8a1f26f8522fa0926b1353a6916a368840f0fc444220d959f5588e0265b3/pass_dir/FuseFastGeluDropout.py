import torch
import triton
import triton.language as tl

# ---------------------------------------------------------------------------
# Fuse conv2d (groups=1, 1×1 kernel) + GELU + dropout(p=0) for fastvit-style
# networks.  The kernel handles:
#   - weight shape [C_out, C_in, 1, 1]  (stride=1, pad=0)
#   - C_in is read-only (no spatial halo needed)
#   - Grid: (N*C_out, ceil(HW/BLOCK_HW))
# ---------------------------------------------------------------------------


def pattern(in_0, in_1, in_2):
    """
    Matches fastvit's:
      conv2d = torch.conv2d(in_2, in_1, in_0, (1,1), (0,0), (1,1), 1)
      tmp_3  = torch.nn.functional.gelu(conv2d, approximate='none')
      tmp_4  = torch.nn.functional.dropout(tmp_3, 0.0, False, False)
    """
    conv2d = torch.conv2d(in_2, in_1, in_0, (1, 1), (0, 0), (1, 1), 1)
    tmp_3  = torch.nn.functional.gelu(conv2d, approximate='none')
    tmp_4  = torch.nn.functional.dropout(tmp_3, 0.0, False, False)
    return tmp_4


def replacement_args(in_0, in_1, in_2):
    return (in_0, in_1, in_2)


# ---------------------------------------------------------------------------
# Triton kernel: 1×1 conv (groups=1) + GELU
#
# Grid:
#   axis-0 : N * C_out  (one program per (batch, output-channel) pair)
#   axis-1 : ceil(H*W / BLOCK_HW)  (spatial tiles)
#
# For each program, accumulates over C_in channels (vectorised in BLOCK_CIN).
# ---------------------------------------------------------------------------

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_HW': 64,  'BLOCK_CIN': 64},  num_warps=4),
        triton.Config({'BLOCK_HW': 128, 'BLOCK_CIN': 64},  num_warps=4),
        triton.Config({'BLOCK_HW': 256, 'BLOCK_CIN': 64},  num_warps=8),
        triton.Config({'BLOCK_HW': 64,  'BLOCK_CIN': 128}, num_warps=4),
        triton.Config({'BLOCK_HW': 128, 'BLOCK_CIN': 128}, num_warps=8),
    ],
    key=['C_out', 'C_in', 'HW'],
)
@triton.jit
def _fastvit_gelu_kernel(
    x_ptr,      # [N, C_in, H, W]
    w_ptr,      # [C_out, C_in, 1, 1]
    b_ptr,      # [C_out]
    out_ptr,    # [N, C_out, H, W]
    N, C_in, C_out, H, W,
    HW,
    BLOCK_HW:  tl.constexpr,
    BLOCK_CIN: tl.constexpr,
):
    nc_out   = tl.program_id(0)   # which (n, c_out) pair
    hw_block = tl.program_id(1)   # which spatial tile

    n     = nc_out // C_out
    c_out = nc_out  % C_out

    # spatial tile
    hw_start = hw_block * BLOCK_HW
    hw_offs  = hw_start + tl.arange(0, BLOCK_HW)
    hw_mask  = hw_offs < HW

    h_v = hw_offs // W
    w_v = hw_offs  % W

    # bias
    b_val = tl.load(b_ptr + c_out).to(tl.float32)
    acc   = tl.zeros([BLOCK_HW], dtype=tl.float32) + b_val

    # accumulate over input channels
    for cin_start in range(0, C_in, BLOCK_CIN):
        cin_offs = cin_start + tl.arange(0, BLOCK_CIN)
        cin_mask = cin_offs < C_in

        # x[n, cin_offs, h_v, w_v]  – stride between channels = HW
        x_off = (n * C_in + cin_offs) * HW + h_v * W + w_v   # [BLOCK_CIN, BLOCK_HW]
        x_valid = cin_mask[:, None] & hw_mask[None, :]
        xv = tl.load(x_ptr + x_off, mask=x_valid, other=0.0).to(tl.float32)

        # w[c_out, cin_offs, 0, 0]  – stride between input channels = 1
        w_off = c_out * C_in + cin_offs
        wv = tl.load(w_ptr + w_off, mask=cin_mask, other=0.0).to(tl.float32)

        # [BLOCK_CIN, BLOCK_HW] × [BLOCK_CIN] -> sum over axis 0 -> [BLOCK_HW]
        acc = acc + tl.sum(xv * wv[:, None], axis=0)

    # GELU: x * 0.5 * (1 + erf(x / √2))
    INV_SQRT2: tl.constexpr = 0.7071067811865476
    gelu = acc * 0.5 * (1.0 + tl.math.erf(acc * INV_SQRT2))

    out_off = (n * C_out + c_out) * HW + hw_offs
    tl.store(out_ptr + out_off, gelu.to(x_ptr.dtype.element_ty), mask=hw_mask)


@torch.fx.wrap
def triton_fastvit_gelu(bias, weight, x):
    """
    bias   : [C_out]
    weight : [C_out, C_in, 1, 1]
    x      : [N, C_in, H, W]
    """
    N, C_in, H, W = x.shape
    C_out = weight.shape[0]
    HW    = H * W

    out  = torch.empty_like(x)
    grid = lambda meta: (N * C_out, triton.cdiv(HW, meta['BLOCK_HW']))

    _fastvit_gelu_kernel[grid](
        x, weight, bias, out,
        N, C_in, C_out, H, W,
        HW,
    )
    return out


def replacement_func():
    return triton_fastvit_gelu