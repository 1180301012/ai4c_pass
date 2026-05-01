import torch
import triton
import triton.language as tl

def pattern(in_0, in_1):
    tmp_0 = torch.sigmoid(in_0)
    tmp_1 = tmp_0.view(1, 512, 1, 1)
    tmp_2 = in_1 * tmp_1
    tmp_3 = in_1 + tmp_2
    return tmp_3

def replacement_args(in_0, in_1):
    return (in_0, in_1)

@triton.jit
def fused_sigmoid_multiply_add_kernel(
    in_0_ptr,
    in_1_ptr,
    out_ptr,
    num_elements,
    BLOCK_SIZE: tl.constexpr,
):
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < num_elements

    # Calculate channel index: each channel spans 64*64 elements
    c = offsets // (64 * 64)
    
    # Load the sigmoid value for this channel (512 elements total)
    sigmoid_val = tl.cast(tl.sigmoid(tl.cast(tl.load(in_0_ptr + c, mask=mask), dtype=tl.float32)), dtype=tl.bfloat16)
    
    # Load input element and compute result
    in_1_val = tl.load(in_1_ptr + offsets, mask=mask)
    result = in_1_val * (1.0 + sigmoid_val)
    
    tl.store(out_ptr + offsets, result, mask=mask)

@torch.fx.wrap
def fused_sigmoid_multiply_add(in_0, in_1):
    n_elements = in_1.numel()
    BLOCK_SIZE = 1024
    num_programs = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    out = torch.empty_like(in_1)
    
    fused_sigmoid_multiply_add_kernel[(num_programs,)](
        in_0_ptr=in_0, 
        in_1_ptr=in_1,
        out_ptr=out,
        num_elements=n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out

def replacement_func():
    return fused_sigmoid_multiply_add