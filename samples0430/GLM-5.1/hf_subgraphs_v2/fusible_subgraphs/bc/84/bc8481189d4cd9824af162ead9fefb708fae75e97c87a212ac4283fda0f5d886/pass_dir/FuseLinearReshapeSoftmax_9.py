import torch
import triton
import triton.language as tl


def pattern(in_0, in_1, in_2):
    linear = torch.nn.functional.linear(in_2, in_1, in_0)
    tmp_3 = torch.reshape(linear, [-1, 9, 1])
    tmp_4 = torch.softmax(tmp_3, dim=1)
    return (tmp_4,)


def replacement_args(in_0, in_1, in_2):
    return (in_0, in_1, in_2)


@triton.jit
def fused_linear_reshape_softmax_kernel(
    in_ptr, weight_ptr, bias_ptr, out_ptr,
    K, S, num_heads_per_row,
    head_dim: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_H: tl.constexpr,
    stride_in_b, stride_in_s, stride_in_k,
    stride_weight_n, stride_weight_k,
    stride_bias_n,
    stride_out_0, stride_out_1,
):
    pid = tl.program_id(0)

    # Each program handles one softmax group
    # Group pid maps to: batch_idx, seq_idx, head_idx
    batch_idx = pid // (S * num_heads_per_row)
    seq_idx = (pid // num_heads_per_row) % S
    head_idx = pid % num_heads_per_row
    out_col_start = head_idx * head_dim

    # Pointer to the input row for this group
    input_row_ptr = in_ptr + batch_idx * stride_in_b + seq_idx * stride_in_s

    # Head offsets and mask (BLOCK_H is power of 2 >= head_dim)
    h_offsets = tl.arange(0, BLOCK_H)
    h_mask = h_offsets < head_dim  # True for indices 0..8, False for 9..15
    w_offsets_n = out_col_start + h_offsets  # Global weight row indices

    # Accumulate dot products in float32 for numerical stability
    acc = tl.zeros([BLOCK_H], dtype=tl.float32)

    # Loop over hidden dimension in tiles
    for k_start in range(0, K, BLOCK_K):
        k_offsets = k_start + tl.arange(0, BLOCK_K)
        k_mask = k_offsets < K

        # Load input row chunk [BLOCK_K]
        x = tl.load(input_row_ptr + k_offsets * stride_in_k, mask=k_mask, other=0.0).to(tl.float32)

        # Load weight block [BLOCK_H, BLOCK_K] for this softmax group
        w_ptrs = weight_ptr + w_offsets_n[:, None] * stride_weight_n + k_offsets[None, :] * stride_weight_k
        w = tl.load(w_ptrs, mask=k_mask[None, :] & h_mask[:, None], other=0.0).to(tl.float32)

        # Compute partial dot products: [BLOCK_H, BLOCK_K] * [BLOCK_K] -> sum over BLOCK_K -> [BLOCK_H]
        acc += tl.sum(w * x[None, :], axis=1)

    # Add bias values (masked load for valid head elements only)
    bias_ptrs = bias_ptr + w_offsets_n * stride_bias_n
    bias_vals = tl.load(bias_ptrs, mask=h_mask, other=0.0).to(tl.float32)
    acc += bias_vals

    # Softmax over the valid head_dim elements (first 9)
    # For max: set padded elements to -inf so they don't affect max
    acc_for_max = tl.where(h_mask, acc, -1e30)
    max_val = tl.max(acc_for_max, axis=0)

    # Exp: only valid elements contribute (padded elements = 0)
    exp_vals = tl.where(h_mask, tl.exp(acc - max_val), 0.0)
    sum_exp = tl.sum(exp_vals, axis=0)

    # Normalize: only valid elements get softmax output
    softmax_out = tl.where(h_mask, exp_vals / sum_exp, 0.0)

    # Store output for valid elements only
    out_ptrs = out_ptr + pid * stride_out_0 + h_offsets * stride_out_1
    tl.store(out_ptrs, softmax_out.to(out_ptr.dtype.element_ty), mask=h_mask)


@torch.fx.wrap
def fused_linear_reshape_softmax(in_0, in_1, in_2):
    # in_0: bias [N], in_1: weight [N, K], in_2: input [B, S, K]
    B = in_2.shape[0]
    S = in_2.shape[1]
    K = in_2.shape[2]
    N = in_1.shape[0]

    head_dim = 9  # from reshape [-1, 9, 1]
    num_heads_per_row = N // head_dim  # 18 // 9 = 2
    num_groups = B * S * num_heads_per_row  # total softmax groups
    BLOCK_H = 16  # Next power of 2 >= head_dim (9)

    # Output shape: [num_groups, head_dim, 1]
    out = torch.empty([num_groups, head_dim, 1], dtype=in_2.dtype, device=in_2.device)

    BLOCK_K = 64

    grid = (num_groups,)

    fused_linear_reshape_softmax_kernel[grid](
        in_ptr=in_2, weight_ptr=in_1, bias_ptr=in_0, out_ptr=out,
        K=K, S=S, num_heads_per_row=num_heads_per_row,
        head_dim=head_dim,
        BLOCK_K=BLOCK_K,
        BLOCK_H=BLOCK_H,
        stride_in_b=in_2.stride(0), stride_in_s=in_2.stride(1), stride_in_k=in_2.stride(2),
        stride_weight_n=in_1.stride(0), stride_weight_k=in_1.stride(1),
        stride_bias_n=in_0.stride(0),
        stride_out_0=out.stride(0), stride_out_1=out.stride(1),
    )

    return (out,)


def replacement_func():
    return fused_linear_reshape_softmax