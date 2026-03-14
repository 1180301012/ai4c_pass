import torch
import triton
import triton.language as tl


# Pattern matching function - matches only the division operation
def pattern(in_1):
    tmp_1 = in_1 / 5.656854249492381
    return tmp_1


# Argument extraction function
def replacement_args(in_1):
    return (in_1,)


# Optimized Triton kernel for division
@triton.jit
def div_kernel(
    in_1_ptr, out_0_ptr,
    in_1_numel: tl.constexpr,
    div_constant: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offset = pid * BLOCK_SIZE
    in_1_offsets = offset + tl.arange(0, BLOCK_SIZE)
    mask = in_1_offsets < in_1_numel
    in_1_vals = tl.load(in_1_ptr + in_1_offsets, mask=mask, other=0.0)
    out_0_vals = in_1_vals / div_constant
    tl.store(out_0_ptr + in_1_offsets, out_0_vals, mask=mask)


@torch.fx.wrap
def optimized_div(in_1):
    out_0_shape = in_1.shape
    out_0 = torch.empty(out_0_shape, dtype=in_1.dtype, device=in_1.device)
    
    div_constant = 5.656854249492381
    in_1_numel = in_1.numel()
    BLOCK_SIZE = 1024
    num_programs = (in_1_numel + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    div_kernel[(num_programs,)](
        in_1_ptr=in_1,
        out_0_ptr=out_0,
        in_1_numel=in_1_numel,
        div_constant=div_constant,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out_0


def replacement_func():
    return optimized_div