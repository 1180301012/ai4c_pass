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
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 64, "BLOCK_K": 32}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 32}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 32, "BLOCK_K": 32}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 128, "BLOCK_K": 32}, num_warps=4, num_stages=2),
    ],
    key=["M", "N"],
)
@triton.jit
def _fused_scale_softmax_matmul_kernel(
    sim_ptr,
    value_ptr,
    out_ptr,
    B,
    M,
    K,
    N,
    stride_sb,
    stride_sm,
    stride_sk,
    stride_vb,
    stride_vk,
    stride_vn,
    stride_ob,
    stride_on,
    stride_om,
    SCALE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid = tl.program_id(0)
    tiles_m = tl.cdiv(M, BLOCK_M)
    tiles_n = tl.cdiv(N, BLOCK_N)
    tiles_per_batch = tiles_m * tiles_n

    b = pid // tiles_per_batch
    pid_in_batch = pid % tiles_per_batch
    pid_n = pid_in_batch // tiles_m
    pid_m = pid_in_batch % tiles_m

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    mask_m = offs_m < M
    mask_n = offs_n < N
    mask_k = offs_k < K

    sim_ptrs = (
        sim_ptr
        + b * stride_sb
        + offs_m[:, None] * stride_sm
        + offs_k[None, :] * stride_sk
    )
    logits = tl.load(sim_ptrs, mask=mask_m[:, None] & mask_k[None, :], other=-float("inf"))
    logits = logits.to(tl.float32) * SCALE

    row_max = tl.max(logits, axis=1)
    logits = logits - row_max[:, None]
    probs = tl.exp(logits)
    denom = tl.sum(probs, axis=1)
    probs = probs / denom[:, None]

    value_ptrs = (
        value_ptr
        + b * stride_vb
        + offs_k[:, None] * stride_vk
        + offs_n[None, :] * stride_vn
    )
    value = tl.load(value_ptrs, mask=mask_k[:, None] & mask_n[None, :], other=0.0)
    value = value.to(tl.float32)

    acc = tl.dot(probs, value)

    out_ptrs = (
        out_ptr
        + b * stride_ob
        + offs_n[None, :] * stride_on
        + offs_m[:, None] * stride_om
    )
    tl.store(out_ptrs, acc, mask=mask_m[:, None] & mask_n[None, :])


@torch.fx.wrap
def fused_scale_softmax_matmul_permute_k19(in_0, in_1):
    B = in_0.shape[0]
    M = in_0.shape[1]
    K = in_0.shape[2]
    N = in_1.shape[2]

    out = torch.empty((B, N, M), device=in_1.device, dtype=in_1.dtype)

    grid = lambda META: (B * triton.cdiv(M, META["BLOCK_M"]) * triton.cdiv(N, META["BLOCK_N"]),)
    _fused_scale_softmax_matmul_kernel[grid](
        in_0,
        in_1,
        out,
        B,
        M,
        K,
        N,
        in_0.stride(0),
        in_0.stride(1),
        in_0.stride(2),
        in_1.stride(0),
        in_1.stride(1),
        in_1.stride(2),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        SCALE=0.0625,
    )
    return out



def replacement_func():
    return fused_scale_softmax_matmul_permute_k19