import torch
import triton
import triton.language as tl


def pattern(in_0, in_1):
    tmp_0 = 0.0625 * in_0
    tmp_1 = torch.nn.functional.softmax(tmp_0, dim=-1)
    matmul = torch.matmul(tmp_1, in_1)
    tmp_3 = matmul.permute(0, 2, 1)
    return tmp_3


def replacement_args(in_0, in_1):
    return (in_0, in_1)


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 32, "BLOCK_K": 64}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 64, "BLOCK_K": 64}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 128, "BLOCK_K": 32}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 32, "BLOCK_K": 128}, num_warps=8, num_stages=2),
    ],
    key=["M", "K"],
)
@triton.jit
def fused_scaled_softmax_bmm_permute_kernel(
    sim_ptr,
    value_ptr,
    out_ptr,
    M,
    K,
    stride_sb,
    stride_sm,
    stride_sn,
    stride_vb,
    stride_vn,
    stride_vk,
    stride_ob,
    stride_ok,
    stride_om,
    N_HEAD: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_k = tl.program_id(1)
    pid_b = tl.program_id(2)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_k = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)

    mask_m = offs_m < M
    mask_k = offs_k < K

    sim_batch_ptr = sim_ptr + pid_b * stride_sb
    value_batch_ptr = value_ptr + pid_b * stride_vb

    row_max = tl.full((BLOCK_M,), -float("inf"), tl.float32)
    for n in range(N_HEAD):
        sim = tl.load(
            sim_batch_ptr + offs_m * stride_sm + n * stride_sn,
            mask=mask_m,
            other=0.0,
        ).to(tl.float32)
        x = sim * 0.0625
        row_max = tl.where(mask_m, tl.maximum(row_max, x), row_max)
    row_max = tl.where(mask_m, row_max, 0.0)

    row_denom = tl.zeros((BLOCK_M,), dtype=tl.float32)
    for n in range(N_HEAD):
        sim = tl.load(
            sim_batch_ptr + offs_m * stride_sm + n * stride_sn,
            mask=mask_m,
            other=0.0,
        ).to(tl.float32)
        x = sim * 0.0625
        w = tl.where(mask_m, tl.exp(x - row_max), 0.0)
        row_denom += w
    row_inv_denom = tl.where(mask_m, 1.0 / row_denom, 0.0)

    acc = tl.zeros((BLOCK_M, BLOCK_K), dtype=tl.float32)
    for n in range(N_HEAD):
        sim = tl.load(
            sim_batch_ptr + offs_m * stride_sm + n * stride_sn,
            mask=mask_m,
            other=0.0,
        ).to(tl.float32)
        x = sim * 0.0625
        w = tl.where(mask_m, tl.exp(x - row_max) * row_inv_denom, 0.0)
        v = tl.load(
            value_batch_ptr + n * stride_vn + offs_k * stride_vk,
            mask=mask_k,
            other=0.0,
        ).to(tl.float32)
        acc += w[:, None] * v[None, :]

    out_ptrs = out_ptr + pid_b * stride_ob + offs_k[:, None] * stride_ok + offs_m[None, :] * stride_om
    out_mask = mask_k[:, None] & mask_m[None, :]
    tl.store(out_ptrs, tl.trans(acc), mask=out_mask)


@torch.fx.wrap
def fused_scaled_softmax_bmm_permute(in_0, in_1):
    batch = in_0.shape[0]
    m = in_0.shape[1]
    n_head = in_0.shape[2]
    k = in_1.shape[2]

    assert in_1.shape[0] == batch
    assert in_1.shape[1] == n_head

    out = torch.empty((batch, k, m), device=in_1.device, dtype=in_1.dtype)

    grid = lambda META: (
        triton.cdiv(m, META["BLOCK_M"]),
        triton.cdiv(k, META["BLOCK_K"]),
        batch,
    )

    fused_scaled_softmax_bmm_permute_kernel[grid](
        in_0,
        in_1,
        out,
        m,
        k,
        in_0.stride(0),
        in_0.stride(1),
        in_0.stride(2),
        in_1.stride(0),
        in_1.stride(1),
        in_1.stride(2),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        N_HEAD=n_head,
    )
    return out


def replacement_func():
    return fused_scaled_softmax_bmm_permute