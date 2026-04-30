import torch
import triton
import triton.language as tl


def pattern(in_0, in_1):
    tmp_2 = torch.nn.functional.embedding(in_0, in_1, 0, None, 2.0, False, False)
    tmp_3 = tmp_2[:, 1:]
    tmp_4 = torch.nn.functional.pad(tmp_3, [0, 0, 0, 1, 0, 0], 'constant', 0.0)
    tmp_5 = tmp_2[:, :-1]
    tmp_6 = torch.nn.functional.pad(tmp_5, [0, 0, 1, 0, 0, 0], 'constant', 0.0)
    tmp_7 = torch.cat([tmp_4, tmp_2, tmp_6], dim=2)
    return tmp_7


def replacement_args(in_0, in_1):
    return (in_0, in_1)


@triton.jit
def fused_embedding_shift_cat_kernel(
    indices_ptr, weight_ptr, output_ptr,
    batch_size, seq_len, vocab_size,
    stride_indices_b, stride_indices_s,
    stride_weight_v, stride_weight_e,
    stride_out_b, stride_out_s, stride_out_e,
    EMBED_DIM: tl.constexpr,
):
    pid = tl.program_id(0)
    i = pid // seq_len
    j = pid % seq_len

    embed_offsets = tl.arange(0, EMBED_DIM)

    # Load indices
    idx_cur = tl.load(indices_ptr + i * stride_indices_b + j * stride_indices_s)

    has_next = (j + 1) < seq_len
    has_prev = j > 0

    if has_next:
        idx_next = tl.load(indices_ptr + i * stride_indices_b + (j + 1) * stride_indices_s)
    if has_prev:
        idx_prev = tl.load(indices_ptr + i * stride_indices_b + (j - 1) * stride_indices_s)

    # Load current embedding
    w_cur = tl.load(weight_ptr + idx_cur * stride_weight_v + embed_offsets * stride_weight_e)
    if idx_cur == 0:  # padding_idx = 0
        w_cur = w_cur - w_cur

    # Load next embedding or zeros
    if has_next:
        w_next = tl.load(weight_ptr + idx_next * stride_weight_v + embed_offsets * stride_weight_e)
        if idx_next == 0:
            w_next = w_next - w_next
    else:
        w_next = w_cur - w_cur

    # Load prev embedding or zeros
    if has_prev:
        w_prev = tl.load(weight_ptr + idx_prev * stride_weight_v + embed_offsets * stride_weight_e)
        if idx_prev == 0:
            w_prev = w_prev - w_prev
    else:
        w_prev = w_cur - w_cur

    # Store output
    base = i * stride_out_b + j * stride_out_s
    tl.store(output_ptr + base + embed_offsets * stride_out_e, w_next)
    tl.store(output_ptr + base + (EMBED_DIM + embed_offsets) * stride_out_e, w_cur)
    tl.store(output_ptr + base + (2 * EMBED_DIM + embed_offsets) * stride_out_e, w_prev)


@torch.fx.wrap
def fused_embedding_shift_cat(indices, weight):
    batch_size, seq_len = indices.shape
    vocab_size, embed_dim = weight.shape
    out_embed_dim = 3 * embed_dim

    output = torch.empty((batch_size, seq_len, out_embed_dim), dtype=weight.dtype, device=weight.device)

    grid = (batch_size * seq_len,)

    fused_embedding_shift_cat_kernel[grid](
        indices_ptr=indices,
        weight_ptr=weight,
        output_ptr=output,
        batch_size=batch_size,
        seq_len=seq_len,
        vocab_size=vocab_size,
        stride_indices_b=indices.stride(0),
        stride_indices_s=indices.stride(1),
        stride_weight_v=weight.stride(0),
        stride_weight_e=weight.stride(1),
        stride_out_b=output.stride(0),
        stride_out_s=output.stride(1),
        stride_out_e=output.stride(2),
        EMBED_DIM=embed_dim,
    )

    return output


def replacement_func():
    return fused_embedding_shift_cat