import torch
import triton
import triton.language as tl

def pattern(in_6, in_5, in_4):
    return torch.nn.functional.linear(in_6, in_5, in_4)

def replacement_args(in_6, in_5, in_4):
    return (in_6, in_5, in_4)

@triton.jit
def linear_element_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    bias_ptr,
    batch_size,
    n,
    k,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    i = pid // n
    j = pid % n
    
    if i >= batch_size or j >= n:
        return
        
    acc = 0.0
    for idx in range(k):
        a_val = tl.load(a_ptr + i * k + idx)
        b_val = tl.load(b_ptr + j * k + idx)
        acc += a_val * b_val
        
    acc += tl.load(bias_ptr + j)
    tl.store(c_ptr + i * n + j, acc)

@torch.fx.wrap
def linear_wrapper(in_6, in_5, in_4):
    B = in_6.shape[0]
    n = in_5.shape[0]
    k = in_5.shape[1]
    out = torch.empty((B, n), dtype=in_6.dtype, device=in_6.device)
    
    grid = ((B * n + 127) // 128, )
    linear_element_kernel[grid](
        a_ptr=in_6,
        b_ptr=in_5,
        c_ptr=out,
        bias_ptr=in_4,
        batch_size=B,
        n=n,
        k=k,
        BLOCK_SIZE=128
    )
    return out

def replacement_func():
    return linear_wrapper