import torch
import triton
import triton.language as tl


def pattern(in_0, relu_out):
    tmp_3 = torch.nn.functional.avg_pool2d(relu_out, 3, 1, 1, False, False, None)
    tmp_4 = tmp_3 - relu_out
    tmp_5 = in_0.unsqueeze(-1)
    tmp_6 = tmp_5.unsqueeze(-1)
    tmp_7 = tmp_6 * tmp_4
    tmp_8 = relu_out + tmp_7
    return tmp_8


def replacement_args(in_0, relu_out):
    return (in_0, relu_out)


@triton.jit
def _fused_pool_scale_kernel(
    in_0_ptr, in_2_ptr, out_ptr,
    C: tl.constexpr, H: tl.constexpr, W: tl.constexpr,
    BLOCK_HW: tl.constexpr,
):
    nc = tl.program_id(0)
    tile_id = tl.program_id(1)
    c = nc % C
    n = nc // C
    HW = H * W
    base = n * (C * HW) + c * HW

    offs = tile_id * BLOCK_HW + tl.arange(0, BLOCK_HW)
    mask = offs < HW
    h = offs // W
    w = offs % W

    scale = tl.load(in_0_ptr + c).to(tl.float32)

    # Input is already relu'd
    center = tl.load(in_2_ptr + base + offs, mask=mask, other=0.0).to(tl.float32)

    # Pool count (count_include_pad=False)
    h_lo = tl.maximum(h - 1, 0)
    h_hi = tl.minimum(h + 2, H)
    w_lo = tl.maximum(w - 1, 0)
    w_hi = tl.minimum(w + 2, W)
    cnt = ((h_hi - h_lo) * (w_hi - w_lo)).to(tl.float32)

    # 3x3 avg pool sum
    psum = tl.zeros([BLOCK_HW], dtype=tl.float32)
    for dh in range(-1, 2):
        for dw in range(-1, 2):
            hh = h + dh
            ww = w + dw
            v = mask & (hh >= 0) & (hh < H) & (ww >= 0) & (ww < W)
            sh = tl.maximum(tl.minimum(hh, H - 1), 0)
            sw = tl.maximum(tl.minimum(ww, W - 1), 0)
            val = tl.load(in_2_ptr + base + sh * W + sw, mask=v, other=0.0).to(tl.float32)
            psum += tl.where(v, val, 0.0)

    pavg = psum / cnt
    res = center + scale * (pavg - center)
    tl.store(out_ptr + base + offs, res, mask=mask)


@torch.fx.wrap
def _fused_efficient_former(in_0, relu_out):
    B = relu_out.shape[0]
    C = relu_out.shape[1]
    H = relu_out.shape[2]
    W = relu_out.shape[3]
    HW = H * W
    BLK = 1024
    nt = (HW + BLK - 1) // BLK

    out0 = torch.empty_like(relu_out)
    _fused_pool_scale_kernel[(B * C, nt)](
        in_0, relu_out, out0,
        C=C, H=H, W=W, BLOCK_HW=BLK,
    )

    return out0


def replacement_func():
    return _fused_efficient_former