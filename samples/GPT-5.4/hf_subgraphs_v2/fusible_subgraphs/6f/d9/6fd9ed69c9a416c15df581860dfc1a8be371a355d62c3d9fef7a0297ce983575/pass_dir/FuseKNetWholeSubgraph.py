import torch
import triton
import triton.language as tl


def pattern(in_0, in_1, in_2, in_3, in_4, in_5):
    linear = torch.nn.functional.linear(in_5, in_1, in_0)
    tmp_5 = linear[(slice(None, None, None), slice(None, 256, None))]
    tmp_6 = tmp_5.view(-1, 256)
    tmp_7 = linear[(slice(None, None, None), slice(-256, None, None))]
    tmp_8 = tmp_7.view(-1, 256)
    tmp_9 = in_4.reshape(300, -1, 256)
    linear_1 = torch.nn.functional.linear(tmp_9, in_3, in_2)
    tmp_11 = linear_1[(Ellipsis, slice(None, 256, None))]
    tmp_12 = linear_1[(Ellipsis, slice(-256, None, None))]
    tmp_13 = tmp_6.unsqueeze(-2)
    return (tmp_11, tmp_12, tmp_8, tmp_13)


def replacement_args(in_0, in_1, in_2, in_3, in_4, in_5):
    return (in_0, in_1, in_2, in_3, in_4, in_5)


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 64, "BLOCK_K": 32}, num_warps=4, num_stages=4),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 32, "BLOCK_K": 32}, num_warps=4, num_stages=4),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 32}, num_warps=4, num_stages=4),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 64}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 32}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_K": 32}, num_warps=8, num_stages=3),
    ],
    key=["M", "N", "K"],
)
@triton.jit
def _fused_dual_linear_split_kernel(
    update_ptr,
    dynamic_weight_ptr,
    dynamic_bias_ptr,
    proposal_ptr,
    input_weight_ptr,
    input_bias_ptr,
    out_dynamic_first_ptr,
    out_dynamic_second_ptr,
    out_input_first_ptr,
    out_input_second_ptr,
    M,
    N,
    K,
    stride_update_m,
    stride_update_k,
    stride_out_dynamic_first_m,
    stride_out_dynamic_first_n,
    stride_out_dynamic_second_m,
    stride_out_dynamic_second_n,
    stride_out_input_first_m,
    stride_out_input_first_n,
    stride_out_input_second_m,
    stride_out_input_second_n,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    out_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)

    acc_dyn_first = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    acc_dyn_second = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k_start in range(0, K, BLOCK_K):
        k_offsets = k_start + offs_k
        a = tl.load(
            update_ptr + offs_m[:, None] * stride_update_m + k_offsets[None, :] * stride_update_k,
            mask=(offs_m[:, None] < M) & (k_offsets[None, :] < K),
            other=0.0,
        )
        w_first = tl.load(
            dynamic_weight_ptr + offs_n[:, None] * K + k_offsets[None, :],
            mask=(offs_n[:, None] < N) & (k_offsets[None, :] < K),
            other=0.0,
        )
        w_second = tl.load(
            dynamic_weight_ptr + (offs_n[:, None] + N) * K + k_offsets[None, :],
            mask=(offs_n[:, None] < N) & (k_offsets[None, :] < K),
            other=0.0,
        )
        acc_dyn_first += tl.dot(a, tl.trans(w_first))
        acc_dyn_second += tl.dot(a, tl.trans(w_second))

    bias_dyn_first = tl.load(dynamic_bias_ptr + offs_n, mask=offs_n < N, other=0.0).to(tl.float32)
    bias_dyn_second = tl.load(dynamic_bias_ptr + N + offs_n, mask=offs_n < N, other=0.0).to(tl.float32)
    acc_dyn_first += bias_dyn_first[None, :]
    acc_dyn_second += bias_dyn_second[None, :]

    tl.store(
        out_dynamic_first_ptr + offs_m[:, None] * stride_out_dynamic_first_m + offs_n[None, :] * stride_out_dynamic_first_n,
        acc_dyn_first,
        mask=out_mask,
    )
    tl.store(
        out_dynamic_second_ptr + offs_m[:, None] * stride_out_dynamic_second_m + offs_n[None, :] * stride_out_dynamic_second_n,
        acc_dyn_second,
        mask=out_mask,
    )

    acc_in_first = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    acc_in_second = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k_start in range(0, K, BLOCK_K):
        k_offsets = k_start + offs_k
        a = tl.load(
            proposal_ptr + offs_m[:, None] * K + k_offsets[None, :],
            mask=(offs_m[:, None] < M) & (k_offsets[None, :] < K),
            other=0.0,
        )
        w_first = tl.load(
            input_weight_ptr + offs_n[:, None] * K + k_offsets[None, :],
            mask=(offs_n[:, None] < N) & (k_offsets[None, :] < K),
            other=0.0,
        )
        w_second = tl.load(
            input_weight_ptr + (offs_n[:, None] + N) * K + k_offsets[None, :],
            mask=(offs_n[:, None] < N) & (k_offsets[None, :] < K),
            other=0.0,
        )
        acc_in_first += tl.dot(a, tl.trans(w_first))
        acc_in_second += tl.dot(a, tl.trans(w_second))

    bias_in_first = tl.load(input_bias_ptr + offs_n, mask=offs_n < N, other=0.0).to(tl.float32)
    bias_in_second = tl.load(input_bias_ptr + N + offs_n, mask=offs_n < N, other=0.0).to(tl.float32)
    acc_in_first += bias_in_first[None, :]
    acc_in_second += bias_in_second[None, :]

    tl.store(
        out_input_first_ptr + offs_m[:, None] * stride_out_input_first_m + offs_n[None, :] * stride_out_input_first_n,
        acc_in_first,
        mask=out_mask,
    )
    tl.store(
        out_input_second_ptr + offs_m[:, None] * stride_out_input_second_m + offs_n[None, :] * stride_out_input_second_n,
        acc_in_second,
        mask=out_mask,
    )


@torch.fx.wrap
def _fused_dual_linear_split_impl(in_0, in_1, in_2, in_3, in_4, in_5):
    m = in_5.shape[0]
    k = in_5.shape[1]
    n = in_1.shape[0] // 2

    out_11 = torch.empty((m, 1, n), device=in_4.device, dtype=in_4.dtype)
    out_12 = torch.empty((m, 1, n), device=in_4.device, dtype=in_4.dtype)
    out_8 = torch.empty((m, n), device=in_5.device, dtype=in_5.dtype)
    out_13 = torch.empty((m, 1, n), device=in_5.device, dtype=in_5.dtype)

    grid = lambda META: (
        triton.cdiv(m, META["BLOCK_M"]),
        triton.cdiv(n, META["BLOCK_N"]),
    )

    _fused_dual_linear_split_kernel[grid](
        in_5,
        in_1,
        in_0,
        in_4,
        in_3,
        in_2,
        out_13,
        out_8,
        out_11,
        out_12,
        m,
        n,
        k,
        in_5.stride(0),
        in_5.stride(1),
        out_13.stride(0),
        out_13.stride(2),
        out_8.stride(0),
        out_8.stride(1),
        out_11.stride(0),
        out_11.stride(2),
        out_12.stride(0),
        out_12.stride(2),
    )

    return (out_11, out_12, out_8, out_13)


def fused_knet_whole_subgraph(in_0, in_1, in_2, in_3, in_4, in_5):
    outs = _fused_dual_linear_split_impl(in_0, in_1, in_2, in_3, in_4, in_5)
    return (outs[0], outs[1], outs[2], outs[3])


def replacement_func():
    return fused_knet_whole_subgraph