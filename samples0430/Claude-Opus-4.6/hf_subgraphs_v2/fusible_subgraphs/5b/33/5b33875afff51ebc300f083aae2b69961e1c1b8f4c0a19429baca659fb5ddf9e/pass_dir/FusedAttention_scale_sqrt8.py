import torch
import triton
import triton.language as tl


def pattern(in_0, in_2, in_3):
    tmp_0 = in_0 / 2.8284271247461903
    tmp_1 = tmp_0 + in_2
    tmp_2 = torch.nn.functional.softmax(tmp_1, dim=-1)
    tmp_3 = torch.nn.functional.dropout(tmp_2, 0.1, False, False)
    matmul = torch.matmul(tmp_3, in_3)
    tmp_5 = matmul.permute(0, 2, 1, 3)
    tmp_6 = tmp_5.contiguous()
    return tmp_6


def replacement_args(in_0, in_2, in_3):
    return (in_0, in_2, in_3)


@triton.jit
def fused_attn_kernel_sqrt8(
    scores_ptr, mask_ptr, value_ptr, out_ptr,
    H: tl.constexpr, S: tl.constexpr, D: tl.constexpr,
    BLOCK_S: tl.constexpr, BLOCK_D: tl.constexpr,
):
    pid = tl.program_id(0)
    b_idx = pid // (H * S)
    remainder = pid % (H * S)
    h_idx = remainder // S
    s_idx = remainder % S

    # Load attention scores row: scores[b, h, s, :]
    s_offsets = tl.arange(0, BLOCK_S)
    scores_offset = b_idx * (H * S * S) + h_idx * (S * S) + s_idx * S
    mask_s = s_offsets < S

    row = tl.load(scores_ptr + scores_offset + s_offsets, mask=mask_s, other=float('-inf')).to(tl.float32)

    # Scale by 1/sqrt(8)
    row = row * 0.3535533905932738

    # Add attention mask [1, 1, 1, S]
    mask_val = tl.load(mask_ptr + s_offsets, mask=mask_s, other=0.0).to(tl.float32)
    row = row + mask_val

    # Softmax
    row_max = tl.max(row, axis=0)
    row = row - row_max
    row_exp = tl.exp(row)
    row_sum = tl.sum(row_exp, axis=0)
    softmax_row = row_exp / row_sum

    # Matmul: output[d] = sum_k softmax[k] * value[b, h, k, d]
    value_base = b_idx * (H * S * D) + h_idx * (S * D)
    d_offsets = tl.arange(0, BLOCK_D)
    k_offsets = tl.arange(0, BLOCK_S)

    value_ptrs = value_ptr + value_base + k_offsets[:, None] * D + d_offsets[None, :]
    v_mask = (k_offsets[:, None] < S) & (d_offsets[None, :] < D)
    value_block = tl.load(value_ptrs, mask=v_mask, other=0.0).to(tl.float32)

    result = tl.sum(softmax_row[:, None] * value_block, axis=0)

    # Store to output[b, s, h, d] (permuted layout)
    out_base = b_idx * (S * H * D) + s_idx * (H * D) + h_idx * D
    out_mask = d_offsets < D
    tl.store(out_ptr + out_base + d_offsets, result.to(out_ptr.dtype.element_ty), mask=out_mask)


@torch.fx.wrap
def fused_attention_sqrt8(in_0, in_2, in_3):
    B, H, S, _ = in_0.shape
    D = in_3.shape[-1]

    BLOCK_S = triton.next_power_of_2(S)
    BLOCK_D = triton.next_power_of_2(D)

    out = torch.empty(B, S, H, D, dtype=in_0.dtype, device=in_0.device)

    grid = (B * H * S,)
    fused_attn_kernel_sqrt8[grid](
        in_0, in_2, in_3, out,
        H, S, D,
        BLOCK_S, BLOCK_D,
    )

    return out


def replacement_func():
    return fused_attention_sqrt8