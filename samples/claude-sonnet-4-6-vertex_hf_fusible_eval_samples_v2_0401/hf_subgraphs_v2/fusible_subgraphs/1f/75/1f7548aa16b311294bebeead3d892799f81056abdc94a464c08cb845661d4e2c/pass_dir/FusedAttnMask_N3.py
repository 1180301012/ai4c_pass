import torch
import triton
import triton.language as tl
from torch import device as _device

@triton.jit
def _mask_combine_N3(
    in_0_ptr,
    in_2_ptr,
    out_ptr,
    B, N,
    stride_b0,
    BLOCK_J: tl.constexpr,
):
    b = tl.program_id(0)
    i = tl.program_id(1)
    in_2_val = tl.load(in_2_ptr + i)
    j_off = tl.arange(0, BLOCK_J)
    j_mask = j_off < N
    in_0_vals = tl.load(in_0_ptr + b * stride_b0 + j_off, mask=j_mask, other=0)
    in_0_bool = in_0_vals != 0
    causal = j_off <= in_2_val
    combined = causal & in_0_bool
    out_idx = b * N * N + i * N + j_off
    tl.store(out_ptr + out_idx, combined.to(tl.int8), mask=j_mask)


@torch.fx.wrap
def _fused_preprocess_N3(in_0, in_1, in_2, in_3):
    B = in_0.shape[0]
    N = 3
    F = in_1.shape[0]
    B3 = in_3.shape[0]
    out_mask = torch.empty(B * N * N, dtype=torch.bool, device=in_0.device)
    _mask_combine_N3[(B, N)](
        in_0, in_2, out_mask,
        B, N,
        in_0.stride(0),
        BLOCK_J=16,
    )
    tmp_13 = out_mask.view(B, 1, N, N)
    tmp_21 = in_1.float().view(1, F, 1)
    tmp_22 = in_3.float().view(B3, 1, N)
    return (tmp_13, tmp_21, tmp_22)


def pattern(in_0, in_1, in_2, in_3):
    tmp_2 = in_0.to(device=_device(type='cuda', index=0), dtype=torch.bool)
    tmp_3 = torch.arange(3, device=_device(type='cuda', index=0))
    tmp_3 += 0
    tmp_5 = tmp_2[slice(None, None, None), tmp_3]
    tmp_6 = torch.arange(3, device=_device(type='cuda', index=0))
    tmp_6 += 0
    tmp_8 = in_2.view(-1, 1)
    tmp_9 = torch.ops.aten.le.Tensor(tmp_6, tmp_8)
    tmp_10 = tmp_9[None, None, slice(None, None, None), slice(None, None, None)]
    tmp_11 = tmp_10.expand(1, -1, -1, -1)
    tmp_12 = tmp_5[slice(None, None, None), None, None, slice(None, None, None)]
    tmp_13 = tmp_11 * tmp_12
    tmp_15 = in_1[None, slice(None, None, None), None]
    tmp_16 = tmp_15.float()
    tmp_17 = tmp_16.expand(1, -1, 1)
    tmp_18 = tmp_17.to(_device(type='cuda', index=0))
    tmp_19 = in_3[slice(None, None, None), None, slice(None, None, None)]
    tmp_20 = tmp_19.float()
    tmp_21 = tmp_18.float()
    tmp_22 = tmp_20.float()
    return (tmp_13, tmp_21, tmp_22)


def replacement_args(in_0, in_1, in_2, in_3):
    return (in_0, in_1, in_2, in_3)


def replacement_func():
    return _fused_preprocess_N3