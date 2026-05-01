import torch
import triton
import triton.language as tl

def pattern(input_tensor):
    tmp0 = input_tensor / 1.6817928305074292
    tmp1 = tmp0.transpose(-1, -2)
    return tmp1

def replacement_args(input_tensor):
    return (input_tensor, 1.6817928305074292)

@triton.jit
def transpose_kernel(
    input_ptr,
    output_ptr,
    S,
    D,
    div_recip,
    BLOCK_SIZE: tl.constexpr
):
    block_x = tl.program_id(0)
    block_y = tl.program_id(1)
    batch = tl.program_id(2)
    
    offs_x = block_x * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    offs_y = block_y * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask_x = offs_x < S
    mask_y = offs_y < D
    
    input_offset = offs_x * D + offs_y
    output_offset = offs_y * S + offs_x
    
    global_input = batch * (S * D) + input_offset
    global_output = batch * (D * S) + output_offset
    
    x = tl.load(input_ptr + global_input, mask=mask_x & mask_y, other=0.0)
    x = x * div_recip
    tl.store(output_ptr + global_output, x, mask=mask_x & mask_y)

@torch.fx.wrap
def kernel_wrapper(input_tensor, divisor):
    B, H, S, D = input_tensor.shape
    div_recip = 1.0 / divisor
    out = torch.empty((B, H, D, S), dtype=input_tensor.dtype, device=input_tensor.device)
    
    BLOCK_SIZE = 32
    grid_x = (S + BLOCK_SIZE - 1) // BLOCK_SIZE
    grid_y = (D + BLOCK_SIZE - 1) // BLOCK_SIZE
    grid_z = B * H
    
    transpose_kernel[(grid_x, grid_y, grid_z)](
        input_tensor,
        out,
        S,
        D,
        div_recip,
        BLOCK_SIZE
    )
    return out

def replacement_func():
    return kernel_wrapper