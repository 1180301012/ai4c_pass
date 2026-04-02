import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_HW': 128, 'BLOCK_C': 64},  num_warps=4),
        triton.Config({'BLOCK_HW': 256, 'BLOCK_C': 64},  num_warps=8),
        triton.Config({'BLOCK_HW': 128, 'BLOCK_C': 128}, num_warps=8),
        triton.Config({'BLOCK_HW': 256, 'BLOCK_C': 32},  num_warps=4),
        triton.Config({'BLOCK_HW': 512, 'BLOCK_C': 32},  num_warps=4),
        triton.Config({'BLOCK_HW': 64,  'BLOCK_C': 128}, num_warps=4),
        triton.Config({'BLOCK_HW': 512, 'BLOCK_C': 64},  num_warps=8),
        triton.Config({'BLOCK_HW': 1024,'BLOCK_C': 32},  num_warps=4),
    ],
    key=['B', 'C', 'HW'],
)
@triton.jit
def _conv1x1_dotprod_b24(
    in2_ptr, w_ptr, bias_ptr, out_ptr,
    B, C, HW,
    BLOCK_HW: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    """Fused 1x1-conv dot-product kernel.
    Grid = (B * ceil(HW/BLOCK_HW),)
    out[b, p] = sum_c(in2[b, c, p] * w[c]) + bias   (float32)
    """
    prog_id = tl.program_id(0)
    num_hw_tiles = (HW + BLOCK_HW - 1) // BLOCK_HW
    b        = prog_id // num_hw_tiles
    hw_tile  = prog_id %  num_hw_tiles

    p_offsets = hw_tile * BLOCK_HW + tl.arange(0, BLOCK_HW)
    p_mask    = p_offsets < HW

    acc = tl.zeros([BLOCK_HW], dtype=tl.float32)

    for c_start in range(0, C, BLOCK_C):
        c_offsets = c_start + tl.arange(0, BLOCK_C)
        c_mask    = c_offsets < C

        # weight  shape [BLOCK_C]
        w = tl.load(w_ptr + c_offsets, mask=c_mask, other=0.0).to(tl.float32)

        # input   shape [BLOCK_C, BLOCK_HW]  (NCHW: in2[b,c,p] = b*C*HW + c*HW + p)
        in2_offs = b * C * HW + c_offsets[:, None] * HW + p_offsets[None, :]
        x = tl.load(in2_ptr + in2_offs,
                    mask=c_mask[:, None] & p_mask[None, :],
                    other=0.0).to(tl.float32)

        acc = acc + tl.sum(x * w[:, None], axis=0)

    bias = tl.load(bias_ptr).to(tl.float32)
    acc  = acc + bias

    tl.store(out_ptr + b * HW + p_offsets, acc, mask=p_mask)


@triton.jit
def _softmax_1d_b24(
    in_ptr, out_ptr,
    HW: tl.constexpr,
):
    """Softmax over HW=4096 for one batch item.  Grid = (B,)"""
    b       = tl.program_id(0)
    offsets = tl.arange(0, HW)

    x = tl.load(in_ptr + b * HW + offsets)          # float32

    x_max   = tl.max(x, axis=0)
    x_exp   = tl.exp(x - x_max)
    x_sum   = tl.sum(x_exp, axis=0)
    result  = x_exp / x_sum

    tl.store(out_ptr + b * HW + offsets, result)


# ─── pattern ───────────────────────────────────────────────────────────────────
def pattern(in_0, in_1, in_2):
    conv   = torch.conv2d(in_2, in_1, in_0, (1, 1), (0, 0), (1, 1), 1)
    view   = conv.view(24, 1, -1)
    result = view.softmax(dim=-1)
    return result


def replacement_args(in_0, in_1, in_2):
    return (in_0, in_1, in_2)


# ─── replacement ───────────────────────────────────────────────────────────────
@torch.fx.wrap
def _fused_b24(in_0, in_1, in_2):
    B  = in_2.shape[0]
    C  = in_2.shape[1]
    HW = in_2.shape[2] * in_2.shape[3]   # 4096

    w_flat = in_1.reshape(-1)             # [C]

    # Phase-1 : compute dot products → float32
    intermediate = torch.empty((B, HW), dtype=torch.float32, device=in_2.device)
    grid = lambda meta: (B * triton.cdiv(HW, meta['BLOCK_HW']),)
    _conv1x1_dotprod_b24[grid](in_2, w_flat, in_0, intermediate, B, C, HW)

    # Phase-2 : softmax → float32
    out_f32 = torch.empty((B, HW), dtype=torch.float32, device=in_2.device)
    _softmax_1d_b24[(B,)](intermediate, out_f32, HW=4096)

    # Cast to input dtype and reshape to [B, 1, HW]
    return out_f32.to(in_2.dtype).view(B, 1, HW)


def replacement_func():
    return _fused_b24