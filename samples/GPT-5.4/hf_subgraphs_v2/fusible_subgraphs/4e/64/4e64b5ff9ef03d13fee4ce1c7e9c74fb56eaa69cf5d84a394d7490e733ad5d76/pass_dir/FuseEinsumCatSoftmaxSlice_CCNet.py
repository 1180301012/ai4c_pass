import torch
import triton
import triton.language as tl


# Pattern matching function
# IMPORTANT: mirror model.py ops/dataflow exactly and return all observable values.
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
        triton.Config({"BLOCK_W": 8, "BLOCK_C": 16}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_W": 8, "BLOCK_C": 32}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_W": 16, "BLOCK_C": 16}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_W": 16, "BLOCK_C": 32}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_W": 32, "BLOCK_C": 16}, num_warps=8, num_stages=2),
    ],
    key=["W", "C"],
)
@triton.jit
def _fused_einsum_cat_softmax_kernel(
    energy_ptr,
    key_ptr,
    query_ptr,
    out_ptr,
    B,
    H,
    W,
    C,
    J,
    energy_sb,
    energy_sh,
    energy_sw,
    energy_sj,
    key_sb,
    key_sc,
    key_sh,
    key_sj,
    query_sb,
    query_sc,
    query_sh,
    query_sw,
    out_sb,
    out_sh,
    out_sw,
    out_sj,
    BLOCK_W: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    pid = tl.program_id(0)
    blocks_per_bh = tl.cdiv(W, BLOCK_W)
    bh_idx = pid // blocks_per_bh
    w_block = pid % blocks_per_bh

    b = bh_idx // H
    h = bh_idx % H

    w_offsets = w_block * BLOCK_W + tl.arange(0, BLOCK_W)
    j_offsets = tl.arange(0, 64)

    acc = tl.zeros((BLOCK_W, 64), dtype=tl.float32)

    c_start = 0
    while c_start < C:
        c_offsets = c_start + tl.arange(0, BLOCK_C)

        q_ptrs = (
            query_ptr
            + b * query_sb
            + c_offsets[:, None] * query_sc
            + h * query_sh
            + w_offsets[None, :] * query_sw
        )
        k_ptrs = (
            key_ptr
            + b * key_sb
            + c_offsets[:, None] * key_sc
            + h * key_sh
            + j_offsets[None, :] * key_sj
        )

        q = tl.load(
            q_ptrs,
            mask=(c_offsets[:, None] < C) & (w_offsets[None, :] < W),
            other=0,
        )
        k = tl.load(
            k_ptrs,
            mask=(c_offsets[:, None] < C),
            other=0,
        )

        acc += tl.dot(tl.trans(q), k)
        c_start += BLOCK_C

    energy_ptrs = (
        energy_ptr
        + b * energy_sb
        + h * energy_sh
        + w_offsets[:, None] * energy_sw
        + j_offsets[None, :] * energy_sj
    )
    energy = tl.load(
        energy_ptrs,
        mask=(w_offsets[:, None] < W),
        other=-float("inf"),
    ).to(tl.float32)

    max_energy = tl.max(energy, axis=1)
    max_acc = tl.max(acc, axis=1)
    row_max = tl.maximum(max_energy, max_acc)

    energy_exp = tl.exp(energy - row_max[:, None])
    acc_exp = tl.exp(acc - row_max[:, None])
    denom = tl.sum(energy_exp, axis=1) + tl.sum(acc_exp, axis=1)

    out_first = energy_exp / denom[:, None]
    out_second = acc_exp / denom[:, None]

    out_ptrs_first = (
        out_ptr
        + b * out_sb
        + h * out_sh
        + w_offsets[:, None] * out_sw
        + j_offsets[None, :] * out_sj
    )
    out_ptrs_second = out_ptrs_first + 64 * out_sj

    row_mask = w_offsets[:, None] < W
    tl.store(out_ptrs_first, out_first, mask=row_mask)
    tl.store(out_ptrs_second, out_second, mask=row_mask)


@torch.fx.wrap
def fused_einsum_cat_softmax_slice(in_0, in_1, in_2):
    # in_0: [B, H, W, 64]
    # in_1: [B, C, H, 64]
    # in_2: [B, C, H, W]
    B = in_0.shape[0]
    H = in_0.shape[1]
    W = in_0.shape[2]
    J = in_0.shape[3]
    C = in_1.shape[1]

    out = torch.empty((B, H, W, J * 2), device=in_0.device, dtype=in_0.dtype)

    grid = lambda META: (B * H * triton.cdiv(W, META["BLOCK_W"]),)

    _fused_einsum_cat_softmax_kernel[grid](
        in_0,
        in_1,
        in_2,
        out,
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
        out.stride(0),
        out.stride(1),
        out.stride(2),
        out.stride(3),
    )

    return out, out[(Ellipsis, slice(None, 64, None))]


# Replacement function (no arguments)
def replacement_func():
    return fused_einsum_cat_softmax_slice