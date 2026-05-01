import torch
import triton
import triton.language as tl

def pattern(in_0, in_1):
    tmp_0 = in_1 + in_0
    tmp_1 = torch.tensor(-3.4028234663852886e+38, device=in_1.device)
    tmp_2 = torch.max(tmp_0, tmp_1)
    tmp_3 = tmp_2.view(16, 13, 13)
    tmp_4 = torch.nn.functional.softmax(tmp_3, dim=-1)
    return tmp_4

def replacement_args(in_0, in_1):
    return (in_0, in_1)

@triton.jit
def fused_masked_softmax_kernel(
    in0_ptr,
    in1_ptr,
    out_ptr,
    heads,
    seq_len,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    head_id = pid // seq_len
    seq_pos = pid % seq_len
    row_start = head_id * seq_len * seq_len + seq_pos * seq_len

    in0_row = tl.load(in0_ptr + row_start + tl.arange(0, seq_len))
    in1_row = tl.load(in1_ptr + row_start + tl.arange(0, seq_len))
    combined = in0_row + in1_row
    inf = -3.4028234663852886e+38
    masked = tl.maximum(combined, inf)
    max_val = tl.max(masked)
    exp_row = tl.exp(masked - max_val)
    sum_exp = tl.sum(exp_row)
    softmax_row = exp_row / sum_exp
    tl.store(out_ptr + row_start + tl.arange(0, seq_len), softmax_row)

@torch.fx.wrap
def fused_masked_softmax(in_0, in_1):
    heads = 16
    seq_len = 13
    num_programs = heads * seq_len
    out = torch.empty((heads, seq_len, seq_len), dtype=in_1.dtype, device=in_1.device)
    fused_masked_softmax_kernel[(num_programs,)](
        in0_ptr=in_0,
        in1_ptr=in_1,
        out_ptr=out,
        heads=heads,
        seq_len=seq_len,
        BLOCK_SIZE=128
    )
    return out

def replacement_func():
    return fused_masked_softmax