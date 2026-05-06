import torch
import triton
import triton.language as tl

def pattern(in_1, in_2):
    return in_1 * torch.nn.functional.relu(in_2)

def replacement_args(in_1, in_2):
    return (in_1, in_2)

@triton.jit
def fused_relu_scale_kernel(
    in_1_ptr,
    in_2_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = (offsets < n_elements)
    in_1 = tl.load(in_1_ptr)
    in_2 = tl.load(in_2_ptr + offsets, mask=mask, other=0.0)
    relu_in_2 = tl.where(in_2 >= 0, in_2, 0.0)
    scaled_in_2 = in_1 * relu_in_2
    tl.store(out_ptr + offsets, scaled_in_2, mask=mask)

@torch.fx.wrap
def fused_relu_scale_kernel_wrapper(in_1, in_2):
    n = in_2.numel()
    block_size = 1024
    num_blocks = (n + block_size - 1) // block_size
    out = torch.empty_like(in_2)
    fused_relu_scale_kernel[(num_blocks,)](
        in_1_ptr=in_1,
        in_2_ptr=in_2,
        out_ptr=out,
        n_elements=n,
        BLOCK_SIZE=block_size
    )
    return out

def replacement_func():
    return fused_relu_scale_kernel_wrapper