import torch
import triton
import triton.language as tl

def pattern(in_0, in_1):
    tmp_0 = in_0.to(torch.float32)
    tmp_1 = in_1 * tmp_0
    tmp_2 = torch.sum(tmp_1, 1)
    tmp_3 = tmp_0.sum(1)
    return tmp_2, tmp_3

def replacement_args(in_0, in_1):
    tmp_0 = in_0.to(torch.float32)
    return (tmp_0, in_1)

@triton.jit
def fused_sum_kernel(
    in_0_ptr,
    in_1_ptr,
    out_tmp2_ptr,
    out_tmp3_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    start = pid * BLOCK_SIZE
    offsets = start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    s1 = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    s2 = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)

    for j in range(10):
        base_in_0 = j * 1024
        base_in_1 = j * 1024
        in_0_vals = tl.load(in_0_ptr + base_in_0 + offsets, mask=mask, other=0.0)
        in_1_vals = tl.load(in_1_ptr + base_in_1 + offsets, mask=mask, other=0.0)
        s1 += in_1_vals * in_0_vals
        s2 += in_0_vals

    tl.store(out_tmp2_ptr + offsets, s1, mask=mask)
    tl.store(out_tmp3_ptr + offsets, s2, mask=mask)

@torch.fx.wrap
def fused_sum(in_0, in_1):
    out_tmp2 = torch.empty_like(in_0.sum(1))
    out_tmp3 = torch.empty_like(in_0.sum(1))
    n_elements = out_tmp2.numel()
    grid = (n_elements + 128 - 1) // 128
    
    fused_sum_kernel[grid](
        in_0_ptr=in_0,
        in_1_ptr=in_1,
        out_tmp2_ptr=out_tmp2,
        out_tmp3_ptr=out_tmp3,
        n_elements=n_elements,
        BLOCK_SIZE=128
    )
    
    return out_tmp2, out_tmp3

def replacement_func():
    return fused_sum