import torch
import triton
import triton.language as tl

def pattern(in_2):
    tmp1 = in_2.float()
    tmp2 = torch.nn.functional.softmax(tmp1, dim=-1)
    tmp3 = tmp2.type_as(in_2)
    tmp4 = torch.nn.functional.dropout(tmp3, p=0.1, training=False)
    return tmp4

def replacement_args(in_2):
    return (in_2,)

@triton.jit
def fused_softmax_dropout_kernel(
    x_ptr,
    out_ptr,
    num_rows: tl.int32,
    S: tl.int32,
    BLOCK_SIZE: tl.constexpr,
):
    block_idx = tl.program_id(0)
    row_start = block_idx * S
    x = tl.load(x_ptr + row_start + tl.arange(0, BLOCK_SIZE), mask=tl.arange(0, BLOCK_SIZE) < S, other=0.0)
    x_max = tl.max(x, axis=0)
    x = x - x_max
    x = tl.exp(x)
    x_sum = tl.sum(x, axis=0)
    y = x / x_sum
    y = y * 0.9  # Dropout scaling for training=False (p=0.1)
    tl.store(out_ptr + row_start + tl.arange(0, BLOCK_SIZE), y, mask=tl.arange(0, BLOCK_SIZE) < S)

@torch.fx.wrap
def fused_softmax_dropout(x):
    B, H, S1, S2 = x.shape
    S = S2  # Last dimension for softmax (dim=-1)
    x_reshaped = x.float().reshape(-1, S)
    out_reshaped = torch.empty_like(x_reshaped)
    grid = (x_reshaped.shape[0], 1)
    BLOCK_SIZE = S
    fused_softmax_dropout_kernel[grid](x_reshaped, out_reshaped, x_reshaped.shape[0], S, BLOCK_SIZE)
    return out_reshaped.reshape(x.shape).type_as(x)

def replacement_func():
    return fused_softmax_dropout