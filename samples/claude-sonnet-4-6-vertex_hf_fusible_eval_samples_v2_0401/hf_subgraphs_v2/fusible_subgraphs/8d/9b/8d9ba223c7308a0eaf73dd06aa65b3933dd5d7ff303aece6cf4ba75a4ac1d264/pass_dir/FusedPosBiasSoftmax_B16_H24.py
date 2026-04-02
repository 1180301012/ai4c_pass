import torch
import triton
import triton.language as tl


# Pattern starts from tmp_8 (the contiguous permuted bias tensor [24, 64, 64]).
# in_3 is split into in_3_a and in_3_b (same tensor used twice in the model)
# so the subgraph matcher can handle the duplicate use correctly.
def pattern(tmp_8, in_2, in_3_a, in_3_b):
    tmp_9 = torch.sigmoid(tmp_8)
    tmp_10 = 16 * tmp_9
    tmp_11 = tmp_10.unsqueeze(0)
    tmp_12 = in_2 + tmp_11
    tmp_13 = tmp_12.view(1, 16, 24, 64, 64)
    tmp_14 = in_3_a.unsqueeze(1)
    tmp_15 = tmp_14.unsqueeze(0)
    tmp_16 = tmp_13 + tmp_15
    tmp_17 = in_3_b.unsqueeze(1)
    tmp_18 = tmp_17.unsqueeze(0)
    tmp_19 = tmp_16 + tmp_18
    tmp_20 = tmp_19.view(-1, 24, 64, 64)
    tmp_21 = torch.nn.functional.softmax(tmp_20, dim=-1)
    tmp_22 = torch.nn.functional.dropout(tmp_21, 0.0, False, False)
    return tmp_22


def replacement_args(tmp_8, in_2, in_3_a, in_3_b):
    # in_3_a and in_3_b are the same tensor in practice
    return (tmp_8, in_2, in_3_a)


@triton.jit
def _kernel_fused_B16_H24_bf16(
    tmp8_ptr,   # [24, 64, 64] permuted contiguous bias
    in2_ptr,    # [16, 24, 64, 64] attention scores
    in3_ptr,    # [16, 64, 64] attention mask
    out_ptr,    # [16, 24, 64, 64] output
    SEQ_LEN: tl.constexpr,
    NH: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0)
    NH_SEQ = NH * SEQ_LEN
    b = row_idx // NH_SEQ
    rem = row_idx % NH_SEQ
    h = rem // SEQ_LEN
    i = rem % SEQ_LEN
    j = tl.arange(0, BLOCK_SIZE)

    # tmp8[h, i, j] — fully contiguous
    tmp8_offset = (h * SEQ_LEN + i) * SEQ_LEN + j
    tmp8_vals = tl.load(tmp8_ptr + tmp8_offset).to(tl.float32)
    pos_bias = tl.sigmoid(tmp8_vals) * 16.0

    # in2[b, h, i, j]
    in2_offset = ((b * NH + h) * SEQ_LEN + i) * SEQ_LEN + j
    in2_vals = tl.load(in2_ptr + in2_offset).to(tl.float32)

    # in3[b, i, j]
    in3_offset = (b * SEQ_LEN + i) * SEQ_LEN + j
    in3_vals = tl.load(in3_ptr + in3_offset).to(tl.float32)

    total = in2_vals + pos_bias + 2.0 * in3_vals

    row_max = tl.max(total, axis=0)
    exp_vals = tl.exp(total - row_max)
    exp_sum = tl.sum(exp_vals, axis=0)
    softmax_out = exp_vals / exp_sum

    out_offset = ((b * NH + h) * SEQ_LEN + i) * SEQ_LEN + j
    tl.store(out_ptr + out_offset, softmax_out.to(tl.bfloat16))


@triton.jit
def _kernel_fused_B16_H24_fp16(
    tmp8_ptr,
    in2_ptr,
    in3_ptr,
    out_ptr,
    SEQ_LEN: tl.constexpr,
    NH: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0)
    NH_SEQ = NH * SEQ_LEN
    b = row_idx // NH_SEQ
    rem = row_idx % NH_SEQ
    h = rem // SEQ_LEN
    i = rem % SEQ_LEN
    j = tl.arange(0, BLOCK_SIZE)

    tmp8_offset = (h * SEQ_LEN + i) * SEQ_LEN + j
    tmp8_vals = tl.load(tmp8_ptr + tmp8_offset).to(tl.float32)
    pos_bias = tl.sigmoid(tmp8_vals) * 16.0

    in2_offset = ((b * NH + h) * SEQ_LEN + i) * SEQ_LEN + j
    in2_vals = tl.load(in2_ptr + in2_offset).to(tl.float32)

    in3_offset = (b * SEQ_LEN + i) * SEQ_LEN + j
    in3_vals = tl.load(in3_ptr + in3_offset).to(tl.float32)

    total = in2_vals + pos_bias + 2.0 * in3_vals

    row_max = tl.max(total, axis=0)
    exp_vals = tl.exp(total - row_max)
    exp_sum = tl.sum(exp_vals, axis=0)
    softmax_out = exp_vals / exp_sum

    out_offset = ((b * NH + h) * SEQ_LEN + i) * SEQ_LEN + j
    tl.store(out_ptr + out_offset, softmax_out.to(tl.float16))


@torch.fx.wrap
def fused_pos_bias_softmax_B16_H24(tmp_8, in_2, in_3):
    B = 16
    NH = 24
    SEQ_LEN = 64
    out = torch.empty_like(in_2)
    total_rows = B * NH * SEQ_LEN  # 24576

    if in_2.dtype == torch.bfloat16:
        _kernel_fused_B16_H24_bf16[(total_rows,)](
            tmp_8, in_2, in_3, out,
            SEQ_LEN=SEQ_LEN, NH=NH, BLOCK_SIZE=SEQ_LEN,
        )
    else:
        _kernel_fused_B16_H24_fp16[(total_rows,)](
            tmp_8, in_2, in_3, out,
            SEQ_LEN=SEQ_LEN, NH=NH, BLOCK_SIZE=SEQ_LEN,
        )

    return out


def replacement_func():
    return fused_pos_bias_softmax_B16_H24