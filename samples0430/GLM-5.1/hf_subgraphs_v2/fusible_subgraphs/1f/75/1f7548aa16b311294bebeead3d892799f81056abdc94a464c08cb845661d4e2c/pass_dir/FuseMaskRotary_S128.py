import torch
import triton
import triton.language as tl
from torch import device


@triton.jit
def causal_mask_kernel(
    in_0_ptr, in_2_ptr, out_ptr,
    B, S,
    stride_in0_0, stride_in0_1,
    stride_in2_0,
    stride_out_0, stride_out_2, stride_out_3,
    BLOCK_J: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_j = tl.program_id(1)
    pid_k = tl.program_id(2)

    j_start = pid_j * BLOCK_J
    k_start = pid_k * BLOCK_K
    j_off = j_start + tl.arange(0, BLOCK_J)
    k_off = k_start + tl.arange(0, BLOCK_K)

    j_mask = j_off < S
    k_mask = k_off < S

    in2_ptrs = in_2_ptr + j_off * stride_in2_0
    in2_vals = tl.load(in2_ptrs, mask=j_mask, other=0)

    in0_ptrs = in_0_ptr + pid_b * stride_in0_0 + k_off * stride_in0_1
    in0_vals = tl.load(in0_ptrs, mask=k_mask, other=0)

    causal = k_off[None, :] <= in2_vals[:, None]
    attn = in0_vals[None, :] != 0
    result = causal & attn

    out_ptrs = out_ptr + pid_b * stride_out_0 + j_off[:, None] * stride_out_2 + k_off[None, :] * stride_out_3
    combined_mask = j_mask[:, None] & k_mask[None, :]
    tl.store(out_ptrs, result, mask=combined_mask)


@triton.jit
def inv_freq_cast_kernel(
    in_1_ptr, out_ptr,
    D,
    stride_in1_0,
    stride_out_1,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    valid = offsets < D

    in1_vals = tl.load(in_1_ptr + offsets * stride_in1_0, mask=valid, other=0.0)
    result = in1_vals.to(tl.float32)

    tl.store(out_ptr + offsets * stride_out_1, result, mask=valid)


@triton.jit
def position_ids_cast_kernel(
    in_3_ptr, out_ptr,
    S,
    stride_in3_0, stride_in3_1,
    stride_out_2,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    valid = offsets < S

    in3_ptrs = in_3_ptr + offsets * stride_in3_1
    in3_vals = tl.load(in3_ptrs, mask=valid, other=0)
    result = in3_vals.to(tl.float32)

    tl.store(out_ptr + offsets * stride_out_2, result, mask=valid)


@torch.fx.wrap
def fused_attention_rotary_dispatch(in_0, in_1, in_2, in_3, route):
    B = in_0.shape[0]
    S = in_0.shape[1]
    D = in_1.shape[0]

    mask_out = torch.empty((B, 1, S, S), dtype=torch.bool, device=in_0.device)
    inv_freq_out = torch.empty((1, D, 1), dtype=torch.float32, device=in_1.device)
    pos_ids_out = torch.empty((1, 1, S), dtype=torch.float32, device=in_3.device)

    BLOCK_J = 32
    BLOCK_K = 32
    grid_mask = (B, (S + BLOCK_J - 1) // BLOCK_J, (S + BLOCK_K - 1) // BLOCK_K)
    causal_mask_kernel[grid_mask](
        in_0, in_2, mask_out,
        B, S,
        in_0.stride(0), in_0.stride(1),
        in_2.stride(0),
        mask_out.stride(0), mask_out.stride(2), mask_out.stride(3),
        BLOCK_J=BLOCK_J,
        BLOCK_K=BLOCK_K,
    )

    BLOCK_SIZE_INV = 64
    grid_inv = ((D + BLOCK_SIZE_INV - 1) // BLOCK_SIZE_INV,)
    inv_freq_cast_kernel[grid_inv](
        in_1, inv_freq_out,
        D,
        in_1.stride(0),
        inv_freq_out.stride(1),
        BLOCK_SIZE=BLOCK_SIZE_INV,
    )

    BLOCK_SIZE_POS = 128
    grid_pos = ((S + BLOCK_SIZE_POS - 1) // BLOCK_SIZE_POS,)
    position_ids_cast_kernel[grid_pos](
        in_3, pos_ids_out,
        S,
        in_3.stride(0), in_3.stride(1),
        pos_ids_out.stride(2),
        BLOCK_SIZE=BLOCK_SIZE_POS,
    )

    return mask_out, inv_freq_out, pos_ids_out


def pattern(in_0, in_1, in_2, in_3):
    tmp_2 = in_0.to(device=device(type='cuda', index=0), dtype=torch.bool)
    tmp_3 = torch.arange(128, device=device(type='cuda', index=0))
    tmp_3 += 0
    tmp_5 = tmp_2[slice(None, None, None), tmp_3]
    tmp_6 = torch.arange(128, device=device(type='cuda', index=0))
    tmp_6 += 0
    tmp_8 = in_2.view(-1, 1)
    tmp_9 = tmp_6 <= tmp_8
    tmp_10 = tmp_9[None, None, slice(None, None, None), slice(None, None, None)]
    tmp_11 = tmp_10.expand(1, -1, -1, -1)
    tmp_12 = tmp_5[slice(None, None, None), None, None, slice(None, None, None)]
    tmp_13 = tmp_11 * tmp_12
    tmp_15 = in_1[None, slice(None, None, None), None]
    tmp_16 = tmp_15.float()
    tmp_17 = tmp_16.expand(1, -1, 1)
    tmp_18 = tmp_17.to(device(type='cuda', index=0))
    tmp_19 = in_3[slice(None, None, None), None, slice(None, None, None)]
    tmp_20 = tmp_19.float()
    tmp_21 = tmp_18.float()
    tmp_22 = tmp_20.float()
    return (tmp_13, tmp_21, tmp_22)


def replacement_args(in_0, in_1, in_2, in_3):
    return (in_0, in_1, in_2, in_3, "S128")


def replacement_func():
    return fused_attention_rotary_dispatch