import torch
from torch import device
import triton
import triton.language as tl


def pattern(in_0, in_1, in_2, in_3):
    tmp_2 = in_0.to(device=device(type='cuda', index=0), dtype=torch.bool)
    tmp_3 = torch.arange(64, device=device(type='cuda', index=0))
    tmp_3 += 0
    tmp_4 = tmp_3
    tmp_5 = tmp_2[(slice(None, None, None), tmp_4)]
    tmp_6 = torch.arange(64, device=device(type='cuda', index=0))
    tmp_6 += 0
    tmp_7 = tmp_6
    tmp_8 = in_2.view(-1, 1)
    tmp_9 = tmp_7 <= tmp_8
    tmp_10 = tmp_9[(None, None, slice(None, None, None), slice(None, None, None))]
    tmp_11 = tmp_10.expand(1, -1, -1, -1)
    tmp_12 = tmp_5[(slice(None, None, None), None, None, slice(None, None, None))]
    tmp_13 = tmp_11 * tmp_12
    _set_grad_enabled = torch.set_grad_enabled(False)
    tmp_15 = in_1[(None, slice(None, None, None), None)]
    tmp_16 = tmp_15.float()
    tmp_17 = tmp_16.expand(1, -1, 1)
    tmp_18 = tmp_17.to(device(type='cuda', index=0))
    tmp_19 = in_3[(slice(None, None, None), None, slice(None, None, None))]
    tmp_20 = tmp_19.float()
    tmp_21 = tmp_18.float()
    tmp_22 = tmp_20.float()
    return (tmp_13, tmp_21, tmp_22)


def replacement_args(in_0, in_1, in_2, in_3):
    return (in_0, in_1, in_2, in_3, "len64")


@triton.jit
def fused_mask_kernel(
    in0_ptr,
    in2_ptr,
    out_ptr,
    B,
    L,
    stride_b,
    stride_l,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_b = tl.program_id(2)

    rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    row_mask = rows < L
    col_mask = cols < L

    pos = tl.load(in2_ptr + rows, mask=row_mask, other=-1)
    attn = tl.load(in0_ptr + pid_b * stride_b + cols * stride_l, mask=col_mask, other=0)

    causal = cols[None, :] <= pos[:, None]
    attn_bool = attn[None, :] != 0
    out = causal & attn_bool

    out_offsets = pid_b * L * L + rows[:, None] * L + cols[None, :]
    out_mask = row_mask[:, None] & col_mask[None, :]
    tl.store(out_ptr + out_offsets, out, mask=out_mask)


@triton.jit
def aux_convert_kernel(
    in1_ptr,
    out1_ptr,
    n1,
    in3_ptr,
    out3_ptr,
    n3,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)

    mask1 = offs < n1
    vals1 = tl.load(in1_ptr + offs, mask=mask1, other=0).to(tl.float32)
    tl.store(out1_ptr + offs, vals1, mask=mask1)

    mask3 = offs < n3
    vals3 = tl.load(in3_ptr + offs, mask=mask3, other=0).to(tl.float32)
    tl.store(out3_ptr + offs, vals3, mask=mask3)


@torch.fx.wrap
def fused_rotary_mask_and_casts(in_0, in_1, in_2, in_3, route):
    B = in_0.shape[0]
    L = in_0.shape[1]
    F = in_1.shape[0]
    P0 = in_3.shape[0]

    out_0 = torch.empty((B, 1, L, L), device=in_0.device, dtype=torch.bool)
    out_1 = torch.empty((1, F, 1), device=in_1.device, dtype=torch.float32)
    out_2 = torch.empty((P0, 1, L), device=in_3.device, dtype=torch.float32)

    if L >= 256:
        block_m = 32
        block_n = 64
        num_warps = 4
    else:
        block_m = 16
        block_n = 64
        num_warps = 4

    stride_b, stride_l = in_0.stride()
    grid = (triton.cdiv(L, block_m), triton.cdiv(L, block_n), B)
    fused_mask_kernel[grid](
        in_0,
        in_2,
        out_0,
        B,
        L,
        stride_b,
        stride_l,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        num_warps=num_warps,
    )

    n1 = F
    n3 = P0 * L
    aux_grid = (triton.cdiv(max(n1, n3), 128),)
    aux_convert_kernel[aux_grid](
        in_1,
        out_1,
        n1,
        in_3,
        out_2,
        n3,
        BLOCK=128,
        num_warps=4,
    )

    if route == 'len2':
        return (out_0, out_1, out_2)
    elif route == 'len3':
        return (out_0, out_1, out_2)
    elif route == 'len64':
        return (out_0, out_1, out_2)
    elif route == 'len128':
        return (out_0, out_1, out_2)
    elif route == 'len256':
        return (out_0, out_1, out_2)
    elif route == 'len512':
        return (out_0, out_1, out_2)
    return (out_0, out_1, out_2)


def replacement_func():
    return fused_rotary_mask_and_casts