import torch
from torch import device
import triton
import triton.language as tl

def pattern(in_0, in_1):
    tmp_0 = in_1 + in_0
    tmp_1 = torch.tensor(-3.4028234663852886e+38, device=device(type='cuda', index=0))
    tmp_2 = torch.max(tmp_0, tmp_1)
    tmp_3 = tmp_2.view(16, 13, 13)
    tmp_4 = torch.nn.functional.softmax(tmp_3, dim=-1)
    tmp_5 = torch.nn.functional.dropout(tmp_4, p=0.1, training=False)
    return tmp_5

def replacement_args(in_0, in_1):
    return (in_0, in_1)

@triton.jit
def softmax_kernel(
    input_ptr,
    output_ptr,
    B, H, S,
    input_stride,
    output_stride,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    row_id = pid
    row_ptr = input_ptr + row_id * input_stride
    row_output = output_ptr + row_id * output_stride
    
    offs = tl.arange(0, S)
    x = tl.load(row_ptr + offs, mask=offs < S, other=0.0)
    
    x_max = tl.max(x, axis=0)
    x = x - x_max
    x = tl.exp(x)
    x_sum = tl.sum(x, axis=0)
    x = x / x_sum
    
    tl.store(row_output + offs, x, mask=offs < S)

@torch.fx.wrap
def optimized_softmax(in_3, dim=-1):
    if dim != -1:
        raise ValueError("Only softmax over last dimension is supported")
    B, H, S = in_3.shape
    input_stride = S
    output_stride = S
    grid = (B * H, )
    
    out = torch.empty_like(in_3)
    softmax_kernel[grid](
        in_3,
        out,
        B, H, S,
        input_stride,
        output_stride,
        S
    )
    return out

def replacement_func():
    return optimized_softmax