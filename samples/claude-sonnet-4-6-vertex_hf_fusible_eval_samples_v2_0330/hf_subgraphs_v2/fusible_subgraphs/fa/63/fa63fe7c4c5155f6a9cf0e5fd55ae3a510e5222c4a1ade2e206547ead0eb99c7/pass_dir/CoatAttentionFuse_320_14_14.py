import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 512}),
        triton.Config({'BLOCK_SIZE': 1024}),
        triton.Config({'BLOCK_SIZE': 2048}),
    ],
    key=['N'],
)
@triton.jit
def _coat_tr_320_14_14(
    in2_ptr, out_ptr,
    N,
    D: tl.constexpr,
    H: tl.constexpr,
    W: tl.constexpr,
    C: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    n_elements = C * H * W
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    c = offsets // (H * W)
    hw = offsets % (H * W)

    head = c // D
    d = c % D
    n_in = hw + 1

    in2_idx = head * (N * D) + n_in * D + d

    val = tl.load(in2_ptr + in2_idx, mask=mask, other=0.0)
    tl.store(out_ptr + offsets, val, mask=mask)


def pattern(in_2):
    tmp_2 = in_2[:, :, 1:, :]
    tmp_3 = tmp_2.transpose(-1, -2)
    tmp_4 = tmp_3.reshape(1, 320, 14, 14)
    tmp_5 = torch.functional.split(tmp_4, [80, 120, 120], dim=1)
    tmp_6 = tmp_5[0]
    tmp_7 = tmp_5[1]
    tmp_8 = tmp_5[2]
    return (tmp_6, tmp_7, tmp_8)


def replacement_args(in_2):
    return (in_2,)


@torch.fx.wrap
def _coat_fused_320_14_14(in_2):
    N = in_2.shape[2]
    D, C, H, W = 40, 320, 14, 14
    out = torch.empty(1, C, H, W, dtype=in_2.dtype, device=in_2.device)

    n_elements = C * H * W
    grid = lambda meta: ((n_elements + meta['BLOCK_SIZE'] - 1) // meta['BLOCK_SIZE'],)
    _coat_tr_320_14_14[grid](in_2, out, N, D=D, H=H, W=W, C=C)

    tmp_6 = out[:, :80, :, :]
    tmp_7 = out[:, 80:200, :, :]
    tmp_8 = out[:, 200:, :, :]
    return (tmp_6, tmp_7, tmp_8)


def replacement_func():
    return _coat_fused_320_14_14