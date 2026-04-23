import torch
import triton
import triton.language as tl
from torch import device

def pattern(in_0):
    tmp_0 = torch.nn.functional.softmax(in_0, dim = 1)
    tmp_1 = torch.linspace(0, 4, steps = 5, device = device(type='cuda', index=0))
    tmp_2 = tmp_0 * tmp_1
    tmp_3 = tmp_2.sum(dim = 1)
    tmp_4 = 5 - tmp_3
    return tmp_4

def replacement_args(in_0):
    return (in_0,)

@triton.jit
def fused_softmax_weighted_sum_kernel(
    in_ptr,
    out_ptr,
    stride_in_row: tl.constexpr,
    N: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < N
    
    # Load input row
    row_start = row_idx * stride_in_row
    x = tl.load(in_ptr + row_start + offsets, mask=mask, other=0.0)
    
    # Compute softmax in float32 for numerical stability
    x_f32 = x.to(tl.float32)
    x_max = tl.max(x_f32, axis=0)
    x_shifted = x_f32 - x_max
    e_x = tl.exp(x_shifted)
    e_x_sum = tl.sum(e_x, axis=0)
    softmax_out = e_x / e_x_sum
    
    # Multiply by linspace weights [0, 1, 2, 3, 4]
    weights = offsets.to(tl.float32)
    weighted = softmax_out * weights
    
    # Sum weighted values
    result = tl.sum(weighted, axis=0)
    
    # 5 - result (N - result)
    out_val = N - result
    
    # Store output
    tl.store(out_ptr + row_idx, out_val)

@torch.fx.wrap
def kernel_wrapper(in_0):
    batch_size = in_0.shape[0]
    N = in_0.shape[1]
    out = torch.empty((batch_size,), dtype=in_0.dtype, device=in_0.device)
    
    BLOCK_SIZE = triton.next_power_of_2(N)
    
    fused_softmax_weighted_sum_kernel[(batch_size,)](
        in_ptr=in_0,
        out_ptr=out,
        stride_in_row=N,
        N=N,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out

def replacement_func():
    return kernel_wrapper