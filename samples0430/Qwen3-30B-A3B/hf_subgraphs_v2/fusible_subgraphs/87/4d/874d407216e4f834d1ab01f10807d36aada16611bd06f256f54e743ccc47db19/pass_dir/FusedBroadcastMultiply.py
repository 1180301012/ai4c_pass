import torch
import triton
import triton.language as tl

def pattern(in_0, in_1, in_2):
    tmp_0 = in_1.view(-1, 1)
    tmp_1 = tmp_0 * in_2
    tmp_2 = in_0.view((-1, 1))
    tmp_3 = tmp_2.expand_as(tmp_1)
    return tmp_3, tmp_1

def replacement_args(in_0, in_1, in_2):
    return (in_0, in_1, in_2)

@triton.jit
def fused_broadcast_mult_kernel(
    in_0_ptr,
    in_1_ptr,
    in_2_ptr,
    out1_ptr,
    out2_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    i = offsets // 16
    j = offsets % 16
    
    in_0_val = tl.load(in_0_ptr + i)
    in_1_val = tl.load(in_1_ptr + i)
    in_2_val = tl.load(in_2_ptr + i * 16 + j, mask=mask)
    
    out1_val = in_1_val * in_2_val
    out2_val = in_0_val
    
    tl.store(out1_ptr + offsets, out1_val, mask=mask)
    tl.store(out2_ptr + offsets, out2_val, mask=mask)

@torch.fx.wrap
def fused_broadcast_mult(in_0, in_1, in_2):
    n_elements = in_0.numel() * in_2.shape[1]
    BLOCK_SIZE = 256
    out1 = torch.empty_like(in_2)
    out2 = torch.empty_like(in_2)
    
    num_programs = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    fused_broadcast_mult_kernel[(num_programs,)](
        in_0,
        in_1,
        in_2,
        out1,
        out2,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )
    return out2, out1

def replacement_func():
    return fused_broadcast_mult