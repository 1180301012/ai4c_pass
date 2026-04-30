import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_K": 256, "BLOCK_J": 1}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_K": 512, "BLOCK_J": 1}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_K": 1024, "BLOCK_J": 1}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_K": 2048, "BLOCK_J": 1}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_K": 4096, "BLOCK_J": 1}, num_warps=8, num_stages=1),
        triton.Config({"BLOCK_K": 256, "BLOCK_J": 2}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_K": 512, "BLOCK_J": 2}, num_warps=8, num_stages=2),
    ],
    key=["N", "J"],
)
@triton.jit
def softmax_expectation_kernel(
    in2_ptr,
    in0_ptr,
    in1_ptr,
    out_softmax_ptr,
    out_cat_ptr,
    N,
    J,
    K,
    stride_in2_n,
    stride_in2_j,
    stride_in2_k,
    stride_in0_k,
    stride_in1_k,
    stride_outs_n,
    stride_outs_j,
    stride_outs_k,
    stride_outc_n,
    stride_outc_j,
    stride_outc_last,
    BLOCK_K: tl.constexpr,
    BLOCK_J: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_j_blk = tl.program_id(1)
    j_offsets = pid_j_blk * BLOCK_J + tl.arange(0, BLOCK_J)
    j_mask = j_offsets < J

    k_offsets = tl.arange(0, BLOCK_K)
    base_in2 = in2_ptr + pid_n * stride_in2_n + j_offsets[:, None] * stride_in2_j

    neg_inf = float("-inf")
    row_max = tl.full([BLOCK_J], neg_inf, tl.float32)
    for k_start in range(0, K, BLOCK_K):
        ks = k_start + k_offsets
        k_mask = ks < K
        vals = tl.load(
            base_in2 + ks[None, :] * stride_in2_k,
            mask=j_mask[:, None] & k_mask[None, :],
            other=neg_inf,
        )
        vals_f32 = vals.to(tl.float32)
        row_max = tl.maximum(row_max, tl.max(vals_f32, axis=1))

    row_sum = tl.zeros([BLOCK_J], tl.float32)
    for k_start in range(0, K, BLOCK_K):
        ks = k_start + k_offsets
        k_mask = ks < K
        vals = tl.load(
            base_in2 + ks[None, :] * stride_in2_k,
            mask=j_mask[:, None] & k_mask[None, :],
            other=neg_inf,
        )
        vals_f32 = vals.to(tl.float32)
        numer = tl.exp(vals_f32 - row_max[:, None])
        numer = tl.where(j_mask[:, None] & k_mask[None, :], numer, 0.0)
        row_sum += tl.sum(numer, axis=1)

    acc_x = tl.zeros([BLOCK_J], tl.float32)
    acc_y = tl.zeros([BLOCK_J], tl.float32)
    out_base = out_softmax_ptr + pid_n * stride_outs_n + j_offsets[:, None] * stride_outs_j
    for k_start in range(0, K, BLOCK_K):
        ks = k_start + k_offsets
        k_mask = ks < K
        vals = tl.load(
            base_in2 + ks[None, :] * stride_in2_k,
            mask=j_mask[:, None] & k_mask[None, :],
            other=neg_inf,
        )
        vals_f32 = vals.to(tl.float32)
        probs = tl.exp(vals_f32 - row_max[:, None]) / row_sum[:, None]
        probs = tl.where(j_mask[:, None] & k_mask[None, :], probs, 0.0)

        kx = ks & 63
        ky = ks >> 6
        x_vals = tl.load(in0_ptr + kx * stride_in0_k, mask=k_mask, other=0.0).to(tl.float32)
        y_vals = tl.load(in1_ptr + ky * stride_in1_k, mask=k_mask, other=0.0).to(tl.float32)
        acc_x += tl.sum(probs * x_vals[None, :], axis=1)
        acc_y += tl.sum(probs * y_vals[None, :], axis=1)

        tl.store(
            out_base + ks[None, :] * stride_outs_k,
            probs,
            mask=j_mask[:, None] & k_mask[None, :],
        )

    out_cat_base = out_cat_ptr + pid_n * stride_outc_n + j_offsets * stride_outc_j
    tl.store(out_cat_base + 0 * stride_outc_last, acc_x, mask=j_mask)
    tl.store(out_cat_base + 1 * stride_outc_last, acc_y, mask=j_mask)


@torch.fx.wrap
def fused_softmax_expectation(in_0, in_1, in_2):
    n = in_2.shape[0]
    j = in_2.shape[1]
    k = in_2.shape[2]

    out_softmax = torch.empty_like(in_2)
    out_cat = torch.empty_like(in_2[:, :, :2])

    block_j = 1
    grid = (n, triton.cdiv(j, block_j))
    softmax_expectation_kernel[grid](
        in_2,
        in_0,
        in_1,
        out_softmax,
        out_cat,
        n,
        j,
        k,
        in_2.stride(0),
        in_2.stride(1),
        in_2.stride(2),
        in_0.stride(3),
        in_1.stride(2),
        out_softmax.stride(0),
        out_softmax.stride(1),
        out_softmax.stride(2),
        out_cat.stride(0),
        out_cat.stride(1),
        out_cat.stride(2),
    )

    return out_softmax.reshape(n, j, 64, 64), out_cat


def replacement_impl(in_0, in_1, in_2):
    return fused_softmax_expectation(in_0, in_1, in_2)