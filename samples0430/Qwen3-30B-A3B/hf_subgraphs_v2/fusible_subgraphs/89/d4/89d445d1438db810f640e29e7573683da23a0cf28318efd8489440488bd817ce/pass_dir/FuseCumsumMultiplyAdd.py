import torch
import triton
import triton.language as tl

def pattern(in_0):
    tmp_1 = torch.cumsum(in_0, dim=1)
    tmp_2 = tmp_1 * in_0
    tmp_3 = tmp_2 - 1
    tmp_4 = tmp_3.long()
    tmp_5 = tmp_4[..., :]
    tmp_6 = tmp_5 + 2
    return tmp_6

def replacement_args(in_0):
    return (in_0,)

@triton.jit
def fused_kernel(
    input_ptr,
    output_ptr,
    B,
    S
):
    batch_idx = tl.program_id(0)
    batch_offset = batch_idx * S
    input_vals = tl.load(input_ptr + batch_offset + tl.arange(0, S), mask=tl.arange(0, S) < S)
    
    cumsum = tl.zeros(S, dtype=tl.int64)
    for i in range(S):
        if i == 0:
            cumsum[i] = input_vals[i]
        else:
            cumsum[i] = cumsum[i-1] + input_vals[i]
    
    output_vals = cumsum * input_vals + 1
    tl.store(output_ptr + batch_offset + tl.arange(0, S), output_vals)

@torch.fx.wrap
def fused_computation(in_0):
    B, S = in_0.shape
    output = torch.empty_like(in_0)
    fused_kernel[(B,)](in_0, output, B, S)
    return output

def replacement_func():
    return fused_computation