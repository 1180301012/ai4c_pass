import torch
import triton
import triton.language as tl


@triton.jit
def _fused_sum_kernel_B128(
    prob_ptr,
    lx_ptr,
    ly_ptr,
    xy_ptr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_SIZE)
    base = pid * BLOCK_SIZE

    x = tl.load(prob_ptr + base + offsets)
    x_f32 = x.to(tl.float32)

    j_idx = offsets % 64
    i_idx = offsets // 64

    lx = tl.load(lx_ptr + j_idx)
    ly = tl.load(ly_ptr + i_idx)

    # Multiply in the native dtype (matching PyTorch mul op), then sum in float32
    sum_x = tl.sum((x * lx).to(tl.float32), axis=0)
    sum_y = tl.sum((x * ly).to(tl.float32), axis=0)

    xy_base = pid * 2
    tl.store(xy_ptr + xy_base,     sum_x.to(x.dtype))
    tl.store(xy_ptr + xy_base + 1, sum_y.to(x.dtype))


@torch.fx.wrap
def _fused_ipr_forward_B128(in_0, in_1, softmax_out):
    # softmax_out: [128, 17, 4096] already computed softmax
    B = 128
    H = 17
    BLOCK_SIZE = 4096
    BH = B * H

    lx = in_0.reshape(-1).contiguous()
    ly = in_1.reshape(-1).contiguous()

    prob_flat = softmax_out.reshape(BH, BLOCK_SIZE)
    xy_flat   = torch.empty(BH * 2, dtype=softmax_out.dtype, device=softmax_out.device)

    _fused_sum_kernel_B128[(BH,)](
        prob_ptr=prob_flat,
        lx_ptr=lx,
        ly_ptr=ly,
        xy_ptr=xy_flat,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    prob_4d = prob_flat.view(B, H, 64, 64)
    xy_2d   = xy_flat.view(B, H, 2)
    return prob_4d, xy_2d


# Non-wrapped wrapper: FX traces INTO this, creating two separate getitem nodes
# so graph_copy returns 2 nodes matching the pattern's 2 returning nodes (tmp_3, tmp_10)
def _replacement_B128(in_0, in_1, softmax_out):
    result = _fused_ipr_forward_B128(in_0, in_1, softmax_out)
    return result[0], result[1]


# Take softmax_out as a wildcard placeholder to avoid matching the softmax
# call_function node (which has ForceArgsTracer normalization issues).
def pattern(in_0, in_1, softmax_out):
    tmp_3  = softmax_out.reshape(-1, 17, 64, 64)
    tmp_4  = tmp_3.mul(in_0)
    tmp_5  = tmp_4.reshape(128, 17, -1)
    tmp_6  = torch.sum(tmp_5, dim=2, keepdim=True)
    tmp_7  = tmp_3.mul(in_1)
    tmp_8  = tmp_7.reshape(128, 17, -1)
    tmp_9  = torch.sum(tmp_8, dim=2, keepdim=True)
    tmp_10 = torch.cat([tmp_6, tmp_9], dim=-1)
    return (tmp_3, tmp_10)


def replacement_args(in_0, in_1, softmax_out):
    return (in_0, in_1, softmax_out)


def replacement_func():
    return _replacement_B128