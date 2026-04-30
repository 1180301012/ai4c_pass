import torch
import triton
import triton.language as tl


def pattern(in_0, in_1, in_2):
    einsum = torch.functional.einsum('bchw,bchj->bhwj', in_2, in_1)
    tmp_2 = torch.cat([in_0, einsum], dim=-1)
    tmp_3 = torch.nn.functional.softmax(tmp_2, dim=-1)
    tmp_4 = tmp_3[(Ellipsis, slice(None, 64, None))]
    return (tmp_3, tmp_4)


def replacement_args(in_0, in_1, in_2):
    return (in_0, in_1, in_2)


@triton.autotune(
    configs=[
        triton.Config({}, num_warps=2, num_stages=2),
        triton.Config({}, num_warps=4, num_stages=2),
        triton.Config({}, num_warps=8, num_stages=2),
        triton.Config({}, num_warps=4, num_stages=4),
    ],
    key=[],
)
@triton.jit
def _fused_einsum_cat_softmax_slice_kernel(
    energy_ptr,
    key_ptr,
    query_ptr,
    out_full_ptr,
    out_slice_ptr,
    B,
    H,
    W,
    C,
    J,
    stride_energy_b,
    stride_energy_h,
    stride_energy_w,
    stride_energy_d,
    stride_key_b,
    stride_key_c,
    stride_key_h,
    stride_key_j,
    stride_query_b,
    stride_query_c,
    stride_query_h,
    stride_query_w,
    stride_out_full_b,
    stride_out_full_h,
    stride_out_full_w,
    stride_out_full_d,
    stride_out_slice_b,
    stride_out_slice_h,
    stride_out_slice_w,
    stride_out_slice_d,
    BLOCK_C: tl.constexpr,
    BLOCK_J: tl.constexpr,
):
    pid = tl.program_id(0)

    w_idx = pid % W
    tmp = pid // W
    h_idx = tmp % H
    b_idx = tmp // H

    c_offsets = tl.arange(0, BLOCK_C)
    j_offsets = tl.arange(0, BLOCK_J)

    query_base = query_ptr + b_idx * stride_query_b + h_idx * stride_query_h + w_idx * stride_query_w
    q = tl.load(query_base + c_offsets * stride_query_c, mask=c_offsets < C, other=0).to(tl.float32)

    key_base = key_ptr + b_idx * stride_key_b + h_idx * stride_key_h
    k_ptrs = key_base + c_offsets[:, None] * stride_key_c + j_offsets[None, :] * stride_key_j
    k = tl.load(k_ptrs, mask=(c_offsets[:, None] < C) & (j_offsets[None, :] < J), other=0).to(tl.float32)
    dots = tl.sum(q[:, None] * k, axis=0)

    energy_base = energy_ptr + b_idx * stride_energy_b + h_idx * stride_energy_h + w_idx * stride_energy_w
    energy = tl.load(energy_base + j_offsets * stride_energy_d, mask=j_offsets < J, other=-float('inf')).to(tl.float32)

    max_energy = tl.max(energy, axis=0)
    max_dots = tl.max(dots, axis=0)
    max_val = tl.maximum(max_energy, max_dots)

    exp_energy = tl.exp(energy - max_val)
    exp_dots = tl.exp(dots - max_val)
    denom = tl.sum(exp_energy, axis=0) + tl.sum(exp_dots, axis=0)

    soft_energy = exp_energy / denom
    soft_dots = exp_dots / denom

    out_full_base = out_full_ptr + b_idx * stride_out_full_b + h_idx * stride_out_full_h + w_idx * stride_out_full_w
    out_slice_base = out_slice_ptr + b_idx * stride_out_slice_b + h_idx * stride_out_slice_h + w_idx * stride_out_slice_w

    tl.store(out_full_base + j_offsets * stride_out_full_d, soft_energy, mask=j_offsets < J)
    tl.store(out_full_base + (J + j_offsets) * stride_out_full_d, soft_dots, mask=j_offsets < J)
    tl.store(out_slice_base + j_offsets * stride_out_slice_d, soft_energy, mask=j_offsets < J)


@torch.fx.wrap
def fused_einsum_cat_softmax_slice(in_0, in_1, in_2):
    B = in_0.shape[0]
    H = in_0.shape[1]
    W = in_0.shape[2]
    J = in_0.shape[3]
    C = in_1.shape[1]

    out_full = torch.empty((B, H, W, J + in_1.shape[3]), device=in_0.device, dtype=in_0.dtype)
    out_slice = torch.empty_like(in_0)

    total_rows = B * H * W

    _fused_einsum_cat_softmax_slice_kernel[(total_rows,)](
        in_0,
        in_1,
        in_2,
        out_full,
        out_slice,
        B,
        H,
        W,
        C,
        J,
        in_0.stride(0),
        in_0.stride(1),
        in_0.stride(2),
        in_0.stride(3),
        in_1.stride(0),
        in_1.stride(1),
        in_1.stride(2),
        in_1.stride(3),
        in_2.stride(0),
        in_2.stride(1),
        in_2.stride(2),
        in_2.stride(3),
        out_full.stride(0),
        out_full.stride(1),
        out_full.stride(2),
        out_full.stride(3),
        out_slice.stride(0),
        out_slice.stride(1),
        out_slice.stride(2),
        out_slice.stride(3),
        BLOCK_C=64,
        BLOCK_J=64,
    )

    return (out_full, out_slice)


def replacement_func():
    return fused_einsum_cat_softmax_slice