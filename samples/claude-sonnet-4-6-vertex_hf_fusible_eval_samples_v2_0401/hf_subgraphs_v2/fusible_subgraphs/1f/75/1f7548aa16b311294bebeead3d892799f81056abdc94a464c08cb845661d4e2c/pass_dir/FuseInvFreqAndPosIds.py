import torch
import triton
import triton.language as tl
from torch import device as _device


@triton.jit
def _cast_inv_freq_kernel(
    in_ptr,     # [F] any float dtype
    out_ptr,    # [1, F, 1] float32
    F,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < F
    x = tl.load(in_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    tl.store(out_ptr + offs, x, mask=mask)


@triton.jit
def _cast_pos_ids_kernel(
    in_ptr,     # [B3, N] int64
    out_ptr,    # [B3, 1, N] float32
    B3, N,
    stride_in_b,
    BLOCK: tl.constexpr,
):
    b = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < N
    x = tl.load(in_ptr + b * stride_in_b + offs, mask=mask, other=0)
    tl.store(out_ptr + b * N + offs, x.to(tl.float32), mask=mask)


@torch.fx.wrap
def _fused_inv_freq_and_pos(in_1, in_3):
    F = in_1.shape[0]
    B3 = in_3.shape[0]
    N = in_3.shape[1]

    # Output for inv_freq: [1, F, 1]
    tmp_21 = torch.empty(F, dtype=torch.float32, device=in_1.device)
    BLOCK_F = max(16, triton.next_power_of_2(F))
    _cast_inv_freq_kernel[(triton.cdiv(F, BLOCK_F),)](
        in_1, tmp_21, F, BLOCK=BLOCK_F,
    )
    tmp_21 = tmp_21.view(1, F, 1)

    # Output for position_ids: [B3, 1, N]
    BLOCK_N = max(16, triton.next_power_of_2(N))
    tmp_22 = torch.empty(B3 * N, dtype=torch.float32, device=in_3.device)
    _cast_pos_ids_kernel[(B3,)](
        in_3, tmp_22, B3, N,
        in_3.stride(0),
        BLOCK=BLOCK_N,
    )
    tmp_22 = tmp_22.view(B3, 1, N)

    return (tmp_21, tmp_22)


def pattern(in_1, in_3):
    tmp_15 = in_1[None, slice(None, None, None), None]
    tmp_16 = tmp_15.float()
    tmp_17 = tmp_16.expand(1, -1, 1)
    tmp_18 = tmp_17.to(_device(type='cuda', index=0))
    tmp_19 = in_3[slice(None, None, None), None, slice(None, None, None)]
    tmp_20 = tmp_19.float()
    tmp_21 = tmp_18.float()
    tmp_22 = tmp_20.float()
    return (tmp_21, tmp_22)


def replacement_args(in_1, in_3):
    return (in_1, in_3)


def replacement_func():
    return _fused_inv_freq_and_pos