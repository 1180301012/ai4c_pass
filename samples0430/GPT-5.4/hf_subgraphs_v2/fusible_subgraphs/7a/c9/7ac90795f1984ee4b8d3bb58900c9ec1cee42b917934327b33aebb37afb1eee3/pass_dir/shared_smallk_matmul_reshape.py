import triton
import triton.language as tl
import torch


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 8}, num_warps=1, num_stages=2),
        triton.Config({"BLOCK_M": 16}, num_warps=1, num_stages=2),
        triton.Config({"BLOCK_M": 32}, num_warps=1, num_stages=2),
        triton.Config({"BLOCK_M": 64}, num_warps=2, num_stages=2),
    ],
    key=["HEAD_DIM", "K"],
)
@triton.jit
def _smallk_batchdot_fp16_kernel(
    in0_ptr,
    in1_ptr,
    out_ptr,
    BATCH,
    HEAD_DIM,
    K,
    stride_in0_b,
    stride_in0_k,
    stride_in1_b,
    stride_in1_m,
    stride_in1_k,
    BLOCK_M: tl.constexpr,
):
    pid = tl.program_id(0)
    offs_m = tl.arange(0, BLOCK_M)
    m_mask = offs_m < HEAD_DIM

    acc = tl.zeros((BLOCK_M,), dtype=tl.float32)
    for kk in tl.static_range(0, 16):
        k_mask = kk < K
        a = tl.load(
            in1_ptr + pid * stride_in1_b + offs_m * stride_in1_m + kk * stride_in1_k,
            mask=m_mask & k_mask,
            other=0.0,
        )
        b = tl.load(
            in0_ptr + pid * stride_in0_b + kk * stride_in0_k,
            mask=k_mask,
            other=0.0,
        )
        acc += a * b

    out_offsets = pid * HEAD_DIM + offs_m
    tl.store(out_ptr + out_offsets, acc.to(tl.float16), mask=m_mask)


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 8}, num_warps=1, num_stages=2),
        triton.Config({"BLOCK_M": 16}, num_warps=1, num_stages=2),
        triton.Config({"BLOCK_M": 32}, num_warps=1, num_stages=2),
        triton.Config({"BLOCK_M": 64}, num_warps=2, num_stages=2),
    ],
    key=["HEAD_DIM", "K"],
)
@triton.jit
def _smallk_batchdot_bf16_kernel(
    in0_ptr,
    in1_ptr,
    out_ptr,
    BATCH,
    HEAD_DIM,
    K,
    stride_in0_b,
    stride_in0_k,
    stride_in1_b,
    stride_in1_m,
    stride_in1_k,
    BLOCK_M: tl.constexpr,
):
    pid = tl.program_id(0)
    offs_m = tl.arange(0, BLOCK_M)
    m_mask = offs_m < HEAD_DIM

    acc = tl.zeros((BLOCK_M,), dtype=tl.float32)
    for kk in tl.static_range(0, 16):
        k_mask = kk < K
        a = tl.load(
            in1_ptr + pid * stride_in1_b + offs_m * stride_in1_m + kk * stride_in1_k,
            mask=m_mask & k_mask,
            other=0.0,
        )
        b = tl.load(
            in0_ptr + pid * stride_in0_b + kk * stride_in0_k,
            mask=k_mask,
            other=0.0,
        )
        acc += a * b

    out_offsets = pid * HEAD_DIM + offs_m
    tl.store(out_ptr + out_offsets, acc.to(tl.bfloat16), mask=m_mask)


@torch.fx.wrap
def smallk_matmul_reshape(in_0, in_1, hidden):
    batch = in_1.shape[0]
    head_dim = in_1.shape[1]
    k = in_1.shape[2]
    rows = (batch * head_dim) // hidden

    out = torch.empty((rows, hidden), device=in_1.device, dtype=in_1.dtype)

    grid = (batch,)
    if in_1.dtype == torch.float16:
        _smallk_batchdot_fp16_kernel[grid](
            in_0,
            in_1,
            out,
            batch,
            head_dim,
            k,
            in_0.stride(0),
            in_0.stride(1),
            in_1.stride(0),
            in_1.stride(1),
            in_1.stride(2),
        )
    else:
        _smallk_batchdot_bf16_kernel[grid](
            in_0,
            in_1,
            out,
            batch,
            head_dim,
            k,
            in_0.stride(0),
            in_0.stride(1),
            in_1.stride(0),
            in_1.stride(1),
            in_1.stride(2),
        )

    return out