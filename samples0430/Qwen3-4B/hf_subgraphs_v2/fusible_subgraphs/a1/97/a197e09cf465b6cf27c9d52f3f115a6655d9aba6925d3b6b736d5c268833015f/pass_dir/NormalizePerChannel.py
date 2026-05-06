import torch
import triton
import triton.language as tl

def pattern(in_0):
    sum_dim = in_0.sum(dim=-1)
    sum_dim = sum_dim.unsqueeze(-1)
    return in_0 / sum_dim

def replacement_args(in_0):
    return (in_0,)

@triton.jit
def normalize_kernel(
    input_ptr,
    output_ptr,
    B: tl.int32,
    C: tl.int32,
    H: tl.int32,
    W: tl.int32,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    h = pid % H
    c = pid // H
    total = tl.zeros(tl.float32)
    for w in tl.arange(0, BLOCK_SIZE):
        idx = (h * W + w)
        val = tl.load(input_ptr + idx)
        total += val
    tl.store(output_ptr + (c * H + h), total)

@torch.fx.wrap
def normalize_wrapper(input):
    B, C, H, W = input.shape
    sum_out = torch.empty((B, C, H), device=input.device, dtype=input.dtype)
    normalize_kernel[(B * C * H,)](
        input_ptr=input,
        output_ptr=sum_out,
        B=B,
        C=C,
        H=H,
        W=W,
        BLOCK_SIZE=1024,
    )
    return input / sum_out.unsqueeze(-1)

def replacement_func():
    return normalize_wrapper