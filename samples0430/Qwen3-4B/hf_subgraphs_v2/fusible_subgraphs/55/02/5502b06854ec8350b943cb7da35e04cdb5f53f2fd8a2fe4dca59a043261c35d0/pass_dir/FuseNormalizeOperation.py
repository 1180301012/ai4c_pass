import torch
import triton
import triton.language as tl

def pattern(in_0, in_1):
    x = in_0.to(torch.float32)
    y = in_1 * x
    s1 = torch.sum(y, dim=1)
    s2 = torch.sum(x, dim=1)
    s2_clamped = torch.clamp(s2, min=1e-09)
    res = s1 / s2_clamped
    result = torch.cat([res], dim=1)
    return result

def replacement_args(in_0, in_1):
    return (in_0, in_1)

@triton.jit
def normalized_kernel(
    in0_ptr: tl.types.pointer[tl.float32],
    in1_ptr: tl.types.pointer[tl.float32],
    out_ptr: tl.types.pointer[tl.float32],
    N: tl.int32,
    C: tl.int32,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    c = pid
    if c >= C:
        return
    
    sum0 = tl.zeros(tl.float32)
    sum_prod = tl.zeros(tl.float32)
    
    for i in range(N):
        offset = i * C + c
        in0_val = tl.load(in0_ptr + offset)
        in1_val = tl.load(in1_ptr + offset)
        sum0 += in0_val
        sum_prod += in1_val * in0_val
     
    clamped_sum0 = tl.maximum(sum0, 1e-09)
    out_val = sum_prod / clamped_sum0
    tl.store(out_ptr + c, out_val)

@torch.fx.wrap
def kernel_wrapper(in_0, in_1):
    N = 10
    C = 1024
    
    out = torch.empty(C, dtype=torch.float32)
    
    BLOCK_SIZE = 256
    normalized_kernel[(tl.cdiv(C, BLOCK_SIZE),)](
        in0_ptr=in_0,
        in1_ptr=in_1,
        out_ptr=out,
        N=N,
        C=C,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    result = torch.cat([out.unsqueeze(0)], dim=1)
    return result

def replacement_func():
    return kernel_wrapper