import triton
import triton.language as tl
import torch


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SEQ": 16, "BLOCK_HIDDEN": 32}, num_warps=2, num_stages=2),
        triton.Config({"BLOCK_SEQ": 16, "BLOCK_HIDDEN": 64}, num_warps=2, num_stages=2),
        triton.Config({"BLOCK_SEQ": 8, "BLOCK_HIDDEN": 128}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_SEQ": 16, "BLOCK_HIDDEN": 128}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_SEQ": 8, "BLOCK_HIDDEN": 256}, num_warps=4, num_stages=2),
    ],
    key=["SEQ", "HIDDEN", "HEADS", "HEAD_DIM", "K"],
)
@triton.jit
def _convbert_matmul_reshape_fp16_kernel(
    in0_ptr,
    in1_ptr,
    out_ptr,
    SEQ,
    HIDDEN,
    HEADS,
    HEAD_DIM,
    K,
    stride_in0_b,
    stride_in0_k,
    stride_in1_b,
    stride_in1_m,
    stride_in1_k,
    stride_out_s,
    stride_out_h,
    BLOCK_SEQ: tl.constexpr,
    BLOCK_HIDDEN: tl.constexpr,
):
    pid_s = tl.program_id(0)
    pid_h = tl.program_id(1)

    offs_s = pid_s * BLOCK_SEQ + tl.arange(0, BLOCK_SEQ)
    offs_h = pid_h * BLOCK_HIDDEN + tl.arange(0, BLOCK_HIDDEN)

    head_idx = offs_h // HEAD_DIM
    m_idx = offs_h % HEAD_DIM
    b_idx = offs_s[:, None] * HEADS + head_idx[None, :]

    acc = tl.zeros((BLOCK_SEQ, BLOCK_HIDDEN), dtype=tl.float32)
    for kk in tl.static_range(0, 16):
        k_mask = kk < K
        a = tl.load(
            in1_ptr + b_idx * stride_in1_b + m_idx[None, :] * stride_in1_m + kk * stride_in1_k,
            mask=(offs_s[:, None] < SEQ) & (offs_h[None, :] < HIDDEN) & k_mask,
            other=0.0,
        )
        b = tl.load(
            in0_ptr + b_idx * stride_in0_b + kk * stride_in0_k,
            mask=(offs_s[:, None] < SEQ) & (offs_h[None, :] < HIDDEN) & k_mask,
            other=0.0,
        )
        acc += a * b

    tl.store(
        out_ptr + offs_s[:, None] * stride_out_s + offs_h[None, :] * stride_out_h,
        acc.to(tl.float16),
        mask=(offs_s[:, None] < SEQ) & (offs_h[None, :] < HIDDEN),
    )


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SEQ": 16, "BLOCK_HIDDEN": 32}, num_warps=2, num_stages=2),
        triton.Config({"BLOCK_SEQ": 16, "BLOCK_HIDDEN": 64}, num_warps=2, num_stages=2),
        triton.Config({"BLOCK_SEQ": 8, "BLOCK_HIDDEN": 128}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_SEQ": 16, "BLOCK_HIDDEN": 128}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_SEQ": 8, "BLOCK_HIDDEN": 256}, num_warps=4, num_stages=2),
    ],
    key=["SEQ", "HIDDEN", "HEADS", "HEAD_DIM", "K"],
)
@triton.jit
def _convbert_matmul_reshape_bf16_kernel(
    in0_ptr,
    in1_ptr,
    out_ptr,
    SEQ,
    HIDDEN,
    HEADS,
    HEAD_DIM,
    K,
    stride_in0_b,
    stride_in0_k,
    stride_in1_b,
    stride_in1_m,
    stride_in1_k,
    stride_out_s,
    stride_out_h,
    BLOCK_SEQ: tl.constexpr,
    BLOCK_HIDDEN: tl.constexpr,
):
    pid_s = tl.program_id(0)
    pid_h = tl.program_id(1)

    offs_s = pid_s * BLOCK_SEQ + tl.arange(0, BLOCK_SEQ)
    offs_h = pid_h * BLOCK_HIDDEN + tl.arange(0, BLOCK_HIDDEN)

    head_idx = offs_h // HEAD_DIM
    m_idx = offs_h % HEAD_DIM
    b_idx = offs_s[:, None] * HEADS + head_idx[None, :]

    acc = tl.zeros((BLOCK_SEQ, BLOCK_HIDDEN), dtype=tl.float32)
    for kk in tl.static_range(0, 16):
        k_mask = kk < K
        a = tl.load(
            in1_ptr + b_idx * stride_in1_b + m_idx[None, :] * stride_in1_m + kk * stride_in1_k,
            mask=(offs_s[:, None] < SEQ) & (offs_h[None, :] < HIDDEN) & k_mask,
            other=0.0,
        )
        b = tl.load(
            in0_ptr + b_idx * stride_in0_b + kk * stride_in0_k,
            mask=(offs_s[:, None] < SEQ) & (offs_h[None, :] < HIDDEN) & k_mask,
            other=0.0,
        )
        acc += a * b

    tl.store(
        out_ptr + offs_s[:, None] * stride_out_s + offs_h[None, :] * stride_out_h,
        acc.to(tl.bfloat16),
        mask=(offs_s[:, None] < SEQ) & (offs_h[None, :] < HIDDEN),
    )


@torch.fx.wrap
def convbert_smallk_batched_matmul_reshape(in_0, in_1, in_2):
    # Replaces:
    #   matmul = torch.matmul(in_1, in_0)
    #   tmp_1 = torch.reshape(matmul, [-1, hidden])
    #   tmp_2 = in_2.transpose(-1, -2)
    # returning (tmp_1, tmp_2)
    #
    # Shape facts for target graphs:
    #   in_1: [heads * seq, head_dim, K]
    #   in_0: [heads * seq, K, 1]
    #   in_2: [1, heads, seq, head_dim]
    #   tmp_1: [seq, heads * head_dim]
    heads = in_2.shape[1]
    seq = in_2.shape[2]
    head_dim = in_1.shape[1]
    hidden = heads * head_dim
    k = in_1.shape[2]

    out = torch.empty((seq, hidden), device=in_1.device, dtype=in_1.dtype)

    grid = lambda META: (
        triton.cdiv(seq, META["BLOCK_SEQ"]),
        triton.cdiv(hidden, META["BLOCK_HIDDEN"]),
    )

    if in_1.dtype == torch.float16:
        _convbert_matmul_reshape_fp16_kernel[grid](
            in_0,
            in_1,
            out,
            seq,
            hidden,
            heads,
            head_dim,
            k,
            in_0.stride(0),
            in_0.stride(1),
            in_1.stride(0),
            in_1.stride(1),
            in_1.stride(2),
            out.stride(0),
            out.stride(1),
        )
    else:
        _convbert_matmul_reshape_bf16_kernel[grid](
            in_0,
            in_1,
            out,
            seq,
            hidden,
            heads,
            head_dim,
            k,
            in_0.stride(0),
            in_0.stride(1),
            in_1.stride(0),
            in_1.stride(1),
            in_1.stride(2),
            out.stride(0),
            out.stride(1),
        )

    tmp_2 = in_2.transpose(-1, -2)
    return out, tmp_2


def convbert_smallk_batched_matmul_reshape_replacement(in_0, in_1, in_2):
    outs = convbert_smallk_batched_matmul_reshape(in_0, in_1, in_2)
    return outs[0], outs[1]