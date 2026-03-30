import torch
import triton
import triton.language as tl


@triton.jit
def _fused_ipr_kernel_B512(
    in2_ptr,
    lx_ptr,
    ly_ptr,
    prob_ptr,
    xy_ptr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_SIZE)
    base = pid * BLOCK_SIZE

    x = tl.load(in2_ptr + base + offsets)
    x_f32 = x.to(tl.float32)
    x_max = tl.max(x_f32, axis=0)
    x_exp = tl.exp(x_f32 - x_max)
    x_sum = tl.sum(x_exp, axis=0)
    probs_f32 = x_exp / x_sum

    tl.store(prob_ptr + base + offsets, probs_f32.to(x.dtype))

    j_idx = offsets % 64
    i_idx = offsets // 64

    lx = tl.load(lx_ptr + j_idx).to(tl.float32)
    ly = tl.load(ly_ptr + i_idx).to(tl.float32)

    sum_x = tl.sum(probs_f32 * lx, axis=0)
    sum_y = tl.sum(probs_f32 * ly, axis=0)

    xy_base = pid * 2
    tl.store(xy_ptr + xy_base,     sum_x.to(x.dtype))
    tl.store(xy_ptr + xy_base + 1, sum_y.to(x.dtype))


@torch.fx.wrap
def _fused_ipr_forward_B512(in_0, in_1, in_2):
    B = 512
    H = 17
    BLOCK_SIZE = 4096
    BH = B * H

    lx = in_0.reshape(-1).contiguous()
    ly = in_1.reshape(-1).contiguous()

    prob_flat = torch.empty(BH * BLOCK_SIZE, dtype=in_2.dtype, device=in_2.device)
    xy_flat   = torch.empty(BH * 2,          dtype=in_2.dtype, device=in_2.device)

    in2_view = in_2.reshape(BH, BLOCK_SIZE)

    _fused_ipr_kernel_B512[(BH,)](
        in2_ptr=in2_view,
        lx_ptr=lx,
        ly_ptr=ly,
        prob_ptr=prob_flat,
        xy_ptr=xy_flat,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    prob_4d = prob_flat.view(B, H, 64, 64)
    xy_2d   = xy_flat.view(B, H, 2)
    return prob_4d, xy_2d


def pattern(in_0, in_1, in_2):
    tmp_2  = torch.nn.functional.softmax(in_2, dim=2)
    tmp_3  = tmp_2.reshape(-1, 17, 64, 64)
    tmp_4  = tmp_3.mul(in_0)
    tmp_5  = tmp_4.reshape(512, 17, -1)
    tmp_6  = torch.sum(tmp_5, dim=2, keepdim=True)
    tmp_7  = tmp_3.mul(in_1)
    tmp_8  = tmp_7.reshape(512, 17, -1)
    tmp_9  = torch.sum(tmp_8, dim=2, keepdim=True)
    tmp_10 = torch.cat([tmp_6, tmp_9], dim=-1)
    return (tmp_3, tmp_10)


def replacement_args(in_0, in_1, in_2):
    return (in_0, in_1, in_2)


def replacement_func():
    return _fused_ipr_forward_B512