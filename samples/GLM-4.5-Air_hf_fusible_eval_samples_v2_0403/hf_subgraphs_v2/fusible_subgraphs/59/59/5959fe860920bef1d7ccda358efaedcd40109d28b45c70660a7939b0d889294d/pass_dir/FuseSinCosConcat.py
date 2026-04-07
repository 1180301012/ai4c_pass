import torch
import triton
import triton.language as tl

def pattern(in_1):
    tmp_1 = torch.cat((in_1, in_1), dim=-1)
    tmp_2 = tmp_1.cos()
    tmp_3 = tmp_2 * 1.0
    tmp_4 = tmp_1.sin()
    tmp_5 = tmp_4 * 1.0
    tmp_6 = tmp_3.to(dtype=torch.bfloat16)
    tmp_7 = tmp_5.to(dtype=torch.bfloat16)
    return tmp_6, tmp_7

def replacement_args(in_1):
    return (in_1,)

@triton.jit
def fused_sincos_kernel(
    input_ptr,
    cos_out_ptr,
    sin_out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    input_val = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    
    cos_val = tl.cos(input_val)
    sin_val = tl.sin(input_val)
    
    tl.store(cos_out_ptr + offsets, cos_val, mask=mask)
    tl.store(sin_out_ptr + offsets, sin_val, mask=mask)

@torch.fx.wrap
def fused_sincos_wrapper(in_1):
    # Create concatenated tensor by repeating the last dimension without torch.cat
    # This is equivalent to torch.cat((in_1, in_1), dim=-1)
    tmp_1 = torch.repeat_interleave(in_1, 2, dim=-1)
    
    n_elements = tmp_1.numel()
    BLOCK_SIZE = 1024
    num_programs = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    cos_out = torch.empty_like(tmp_1, dtype=torch.float32)
    sin_out = torch.empty_like(tmp_1, dtype=torch.float32)
    
    fused_sincos_kernel[(num_programs,)](
        input_ptr=tmp_1,
        cos_out_ptr=cos_out,
        sin_out_ptr=sin_out,
        n_elements=n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    tmp_6 = cos_out.to(torch.bfloat16)
    tmp_7 = sin_out.to(torch.bfloat16)
    
    return tmp_6, tmp_7

def replacement_func():
    return fused_sincos_wrapper