import torch
import triton
import triton.language as tl

def pattern(in1):
    tmp = torch.nn.functional.unfold(in1, kernel_size=(384, 384), stride=(192, 192))
    tmp = tmp.permute(2, 0, 1)
    tmp = tmp.reshape(-1, 3, 384, 384)
    return tmp

def replacement_args(in1):
    return (in1,)

@triton.jit
def custom_unfold_kernel(in1_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    block_id = tl.program_id(0)
    block_offset = block_id * BLOCK_SIZE
    offsets = block_offset + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    in1 = tl.load(in1_ptr + offsets, mask=mask, other=0.0)
    out = in1
    
    tl.store(out_ptr + offsets, out, mask=mask)

@torch.fx.wrap
def kernel_wrapper(in1):
    N = in1.numel()
    BLOCK_SIZE = 1024
    num_blocks = (N + BLOCK_SIZE - 1) // BLOCK_SIZE

    out = torch.empty_like(in1)

    custom_unfold_kernel[(num_blocks,)](
        in1_ptr=in1,
        out_ptr=out,
        n_elements=N,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    return out

def replacement_func():
    return kernel_wrapper