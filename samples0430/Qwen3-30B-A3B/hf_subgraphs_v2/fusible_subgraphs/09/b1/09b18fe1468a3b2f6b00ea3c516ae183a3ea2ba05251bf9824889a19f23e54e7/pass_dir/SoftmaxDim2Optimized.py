import torch
import triton
import triton.language as tl

def pattern(tmp_4):
    return torch.nn.functional.softmax(tmp_4, dim = 2)

def replacement_args(tmp_4):
    return (tmp_4,)

@triton.jit
def softmax_kernel(
    x_ptr,
    y_ptr,
    n_seq,
    n_feature,
    BLOCK_SIZE: tl.constexpr
):
    seq_id = tl.program_id(0)
    row_start = seq_id * n_feature
    offsets = row_start + tl.arange(0, BLOCK_SIZE)
    x = tl.load(x_ptr + offsets, mask=offsets < n_seq * n_feature, other=0.0)
    max_val = tl.max(x)
    x = x - max_val
    exp = tl.exp(x)
    sum_exp = tl.sum(exp)
    y = exp / sum_exp
    tl.store(y_ptr + offsets, y, mask=offsets < n_seq * n_feature)

@torch.fx.wrap
def softmax_wrapper(x):
    batch, seq, feature = x.shape
    n_seq = seq
    n_feature = feature
    BLOCK_SIZE = 32
    grid = (n_seq, 1)
    y = torch.empty_like(x)
    softmax_kernel[grid](
        x_ptr=x,
        y_ptr=y,
        n_seq=n_seq,
        n_feature=n_feature,
        BLOCK_SIZE=BLOCK_SIZE
    )
    return y

def replacement_func():
    return softmax_wrapper