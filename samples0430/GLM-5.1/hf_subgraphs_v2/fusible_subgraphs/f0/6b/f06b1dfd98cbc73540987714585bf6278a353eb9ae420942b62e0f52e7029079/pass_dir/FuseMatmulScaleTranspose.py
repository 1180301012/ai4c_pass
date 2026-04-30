import torch
import triton
import triton.language as tl


def pattern(in_0, in_1, in_2):
    matmul = torch.matmul(in_2, in_1)
    tmp_1 = matmul * in_0
    tmp_2 = tmp_1.t()
    return (tmp_1, tmp_2)


def replacement_args(in_0, in_1, in_2):
    return (in_0, in_1, in_2)


@triton.jit
def fused_matmul_scale_trans_kernel(
    in_0_ptr,
    in_1_ptr,
    in_2_ptr,
    out1_ptr,
    out2_ptr,
    M,
    K,
    N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    row_idx = tl.program_id(0)

    # Load scalar scale factor
    scale = tl.load(in_0_ptr).to(tl.float32)

    # Accumulate dot product for this row across all N columns
    n_offsets = tl.arange(0, N)
    acc = tl.zeros([N], dtype=tl.float32)

    for k_start in range(0, K, BLOCK_K):
        k_offsets = k_start + tl.arange(0, BLOCK_K)
        k_mask = k_offsets < K

        # Load row segment from in_2 [M, K] -> [BLOCK_K]
        a = tl.load(in_2_ptr + row_idx * K + k_offsets, mask=k_mask, other=0.0).to(tl.float32)

        # Load from in_1 [K, N] -> [BLOCK_K, N]
        b_ptrs = in_1_ptr + k_offsets[:, None] * N + n_offsets[None, :]
        b = tl.load(b_ptrs, mask=k_mask[:, None], other=0.0).to(tl.float32)

        # Element-wise multiply and reduce over K dimension
        acc += tl.sum(a[:, None] * b, axis=0)

    # Apply scalar multiplication
    result = acc * scale

    # Store to out1 [M, N] - original (non-transposed) result
    tl.store(out1_ptr + row_idx * N + n_offsets, result)

    # Store to out2 [N, M] - transposed result
    tl.store(out2_ptr + n_offsets * M + row_idx, result)


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_K": 64}, num_warps=2),
        triton.Config({"BLOCK_K": 128}, num_warps=2),
        triton.Config({"BLOCK_K": 256}, num_warps=2),
        triton.Config({"BLOCK_K": 512}, num_warps=2),
        triton.Config({"BLOCK_K": 64}, num_warps=4),
        triton.Config({"BLOCK_K": 128}, num_warps=4),
        triton.Config({"BLOCK_K": 256}, num_warps=4),
        triton.Config({"BLOCK_K": 512}, num_warps=4),
    ],
    key=["K"],
)
@triton.jit
def fused_matmul_scale_trans_kernel_autotuned(
    in_0_ptr,
    in_1_ptr,
    in_2_ptr,
    out1_ptr,
    out2_ptr,
    M,
    K,
    N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    row_idx = tl.program_id(0)

    # Load scalar scale factor
    scale = tl.load(in_0_ptr).to(tl.float32)

    # Accumulate dot product for this row across all N columns
    n_offsets = tl.arange(0, N)
    acc = tl.zeros([N], dtype=tl.float32)

    for k_start in range(0, K, BLOCK_K):
        k_offsets = k_start + tl.arange(0, BLOCK_K)
        k_mask = k_offsets < K

        # Load row segment from in_2 [M, K] -> [BLOCK_K]
        a = tl.load(in_2_ptr + row_idx * K + k_offsets, mask=k_mask, other=0.0).to(tl.float32)

        # Load from in_1 [K, N] -> [BLOCK_K, N]
        b_ptrs = in_1_ptr + k_offsets[:, None] * N + n_offsets[None, :]
        b = tl.load(b_ptrs, mask=k_mask[:, None], other=0.0).to(tl.float32)

        # Element-wise multiply and reduce over K dimension
        acc += tl.sum(a[:, None] * b, axis=0)

    # Apply scalar multiplication
    result = acc * scale

    # Store to out1 [M, N] - original (non-transposed) result
    tl.store(out1_ptr + row_idx * N + n_offsets, result)

    # Store to out2 [N, M] - transposed result
    tl.store(out2_ptr + n_offsets * M + row_idx, result)


@torch.fx.wrap
def fused_matmul_scale_trans(in_0, in_1, in_2):
    M = in_2.shape[0]
    K = in_2.shape[1]
    N = in_1.shape[1]

    dtype = in_2.dtype

    out1 = torch.empty((M, N), dtype=dtype, device=in_2.device)
    out2 = torch.empty((N, M), dtype=dtype, device=in_2.device)

    grid = (M,)

    fused_matmul_scale_trans_kernel_autotuned[grid](
        in_0_ptr=in_0,
        in_1_ptr=in_1,
        in_2_ptr=in_2,
        out1_ptr=out1,
        out2_ptr=out2,
        M=M,
        K=K,
        N=N,
    )

    return out1, out2


def replacement_func():
    return fused_matmul_scale_trans