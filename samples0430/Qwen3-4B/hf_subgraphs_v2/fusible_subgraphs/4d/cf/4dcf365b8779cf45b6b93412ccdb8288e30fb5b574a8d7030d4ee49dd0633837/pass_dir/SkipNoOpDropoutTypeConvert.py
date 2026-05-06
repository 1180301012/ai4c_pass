import torch
import triton
import triton.language as tl

def pattern(in_0, in_1, in_2):
    dropout_out = torch.nn.functional.dropout(in_2, p=0.0, training=False)
    converted_out = dropout_out.to(torch.float16)
    linear_out = torch.nn.functional.linear(converted_out, in_1, in_0)
    return linear_out

def replacement_args(in_0, in_1, in_2):
    return (in_0, in_1, in_2)

@triton.jit
def triton_linear_kernel(
    in_ptr,
    weight_ptr,
    bias_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    in_vals = tl.load(in_ptr + offsets, mask=mask, other=0.0)
    weight_vals = tl.load(weight_ptr + offsets, mask=mask, other=0.0)
    bias_vals = tl.load(bias_ptr + offsets, mask=mask, other=0.0)
    
    out_vals = in_vals * weight_vals + bias_vals
    tl.store(out_ptr + offsets, out_vals, mask=mask)

@torch.fx.wrap
def kernel_wrapper(in_0, in_1, in_2):
    n_elements = in_2.numel()
    BLOCK_SIZE = 128
    out = torch.empty_like(in_2)
    
    triton_linear_kernel[(n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE, ](
        in_ptr=in_2,
        weight_ptr=in_1,
        bias_ptr=in_0,
        out_ptr=out,
        n_elements=n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return out

def replacement_func():
    return kernel_wrapper