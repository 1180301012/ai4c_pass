import torch
import triton
import triton.language as tl

_DIM0 = 12
_DIM2 = 9


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_HW': 128, 'BLOCK_CO': 16, 'BLOCK_K': 64},  num_warps=4, num_stages=3),
        triton.Config({'BLOCK_HW': 64,  'BLOCK_CO': 16, 'BLOCK_K': 64},  num_warps=4, num_stages=3),
        triton.Config({'BLOCK_HW': 128, 'BLOCK_CO': 16, 'BLOCK_K': 128}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_HW': 256, 'BLOCK_CO': 16, 'BLOCK_K': 64},  num_warps=8, num_stages=3),
    ],
    key=['C_in', 'C_out', 'HW'],
)
@triton.jit
def _k1x1_sigmoid_12_9(
    x_ptr, w_ptr, b_ptr, out_ptr,
    C_in, C_out, HW,
    BLOCK_HW: tl.constexpr,
    BLOCK_CO: tl.constexpr,
    BLOCK_K:  tl.constexpr,
):
    pid_n  = tl.program_id(0)
    pid_hw = tl.program_id(1)
    pid_co = tl.program_id(2)
    hw_offs = pid_hw * BLOCK_HW + tl.arange(0, BLOCK_HW)
    co_offs = pid_co * BLOCK_CO + tl.arange(0, BLOCK_CO)
    hw_mask = hw_offs < HW
    co_mask = co_offs < C_out
    acc = tl.zeros([BLOCK_HW, BLOCK_CO], dtype=tl.float32)
    for k0 in range(0, C_in, BLOCK_K):
        k_offs = k0 + tl.arange(0, BLOCK_K)
        k_mask = k_offs < C_in
        xt = tl.load(x_ptr + pid_n * C_in * HW + k_offs[:, None] * HW + hw_offs[None, :],
                     mask=k_mask[:, None] & hw_mask[None, :], other=0.0)
        wt = tl.load(w_ptr + co_offs[:, None] * C_in + k_offs[None, :],
                     mask=co_mask[:, None] & k_mask[None, :], other=0.0)
        acc = tl.dot(tl.trans(xt), tl.trans(wt), acc)
    b = tl.load(b_ptr + co_offs, mask=co_mask, other=0.0)
    acc += b[None, :]
    out_val = tl.sigmoid(acc).to(xt.dtype)
    tl.store(out_ptr + pid_n * HW * C_out + hw_offs[:, None] * C_out + co_offs[None, :],
             out_val, mask=hw_mask[:, None] & co_mask[None, :])


@torch.fx.wrap
def _fused_conv_permute_sigmoid_12_9(bias, weight, inp):
    N, C_in, H, W = inp.shape[0], inp.shape[1], inp.shape[2], inp.shape[3]
    C_out = weight.shape[0]
    HW = H * W
    out = torch.empty((N, HW, C_out), dtype=inp.dtype, device=inp.device)
    w_flat = weight.view(C_out, C_in)
    grid = lambda meta: (N, triton.cdiv(HW, meta['BLOCK_HW']), triton.cdiv(C_out, meta['BLOCK_CO']))
    _k1x1_sigmoid_12_9[grid](inp, w_flat, bias, out, C_in, C_out, HW)
    return (out,)


def pattern(bias, weight, inp):
    conv2d = torch.conv2d(inp, weight, bias, (1, 1), (0, 0), (1, 1), 1)
    tmp_3  = conv2d.permute(0, 2, 3, 1)
    tmp_4  = tmp_3.reshape(_DIM0, -1, _DIM2)
    tmp_5  = torch.nn.functional.sigmoid(tmp_4)
    return (tmp_5,)


def replacement_args(bias, weight, inp):
    return (bias, weight, inp)


def replacement_func():
    return _fused_conv_permute_sigmoid_12_9