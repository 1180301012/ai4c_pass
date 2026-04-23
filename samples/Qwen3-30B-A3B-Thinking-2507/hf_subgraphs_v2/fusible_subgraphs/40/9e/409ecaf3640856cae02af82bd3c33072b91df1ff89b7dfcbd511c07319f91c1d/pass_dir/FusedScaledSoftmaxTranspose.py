import torch
import triton
import triton.language as tl

def pattern(in_0):
    tmp_0 = in_0 * 0.1767766952966369
    tmp_1 = tmp_0.softmax(dim=-1)
    tmp_2 = tmp_1.transpose(-2, -1)
    return tmp_2

def replacement_args(in_0):
    return (in_0,)

@triton.jit
def fused_scaled_softmax_transpose_kernel(
    input_ptr,
    output_ptr,
    B: tl.constexpr,
    H: tl.constexpr,
    S: tl.constexpr,
    scale: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    batch_id = tl.program_id(0)
    b = batch_id // (H * S)
    h = (batch_id % (H * S)) // S
    i = batch_id % S
    row_start = b * (H * S * S) + h * (S * S) + i * S
    x = tl.load(input_ptr + row_start, mask=tl.arange(0, S) < S, other=0.0)
    x_scaled = x * scale
    x_max = tl.max(x_scaled)
    x_exp = tl.exp(x_scaled - x_max)
    x_sum = tl.sum(x_exp)
    x_softmax = x_exp / x_sum
    output_start = b * (H * S * S) + h * (S * S)
    output_indices = output_start + tl.arange(0, S) * S + i
    tl.store(output_ptr + output_indices, x_softmax, mask=tl.arange(0, S) < S)

@torch.fx.wrap
def fused_scaled_softmax_transpose(in_0):
    B, H, S, _ = in_0.shape
    num_blocks = B * H * S
    out = torch.empty_like(in_0)
    fused_scaled_softmax_transpose_kernel[(num_blocks, 1)](
        in_0,
        out,
        B, H, S, 0.1767766952966369, S
    )
    return out

def replacement_func():
    return fused_scaled_softmax_transpose