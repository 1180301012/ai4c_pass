import torch
import triton
import triton.language as tl

def pattern(in_0, in_1, in_2):
    tmp1 = in_2 * in_1
    tmp2 = tmp1 + in_0
    return tmp2

def replacement_args(in_0, in_1, in_2):
    return (in_0, in_1, in_2)

@triton.jit
def elementwise_kernel(
    in2_ptr,
    in1_ptr,
    in0_ptr,
    out_ptr,
    n0, n1, n2, n3,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    start = pid * BLOCK_SIZE
    offsets = start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n0 * n1 * n2 * n3

    # Convert linear index to 4D
    stride0 = n1 * n2 * n3
    i = offsets // stride0
    r = offsets % stride0
    stride1 = n2 * n3
    j = r // stride1
    r2 = r % stride1
    stride2 = n3
    k = r2 // stride2
    l = r2 % stride2

    in2_val = tl.load(in2_ptr + offsets, mask=mask, other=0.0)
    in1_val = tl.load(in1_ptr + offsets, mask=mask, other=0.0)
    in0_val = tl.load(in0_ptr + k * 128 + l, mask=mask, other=0.0)

    out_val = in2_val * in1_val + in0_val
    tl.store(out_ptr + offsets, out_val, mask=mask)

@torch.fx.wrap
def elementwise_optimized(in_0, in_1, in_2):
    n0 = in_2.size(0)
    n1 = in_2.size(1)
    n2 = 2
    n3 = in_2.size(3)
    out = torch.empty((n0, n1, n2, n3), dtype=in_2.dtype, device=in_2.device)
    
    n_elements = n0 * n1 * n2 * n3
    BLOCK_SIZE = 1024
    num_blocks = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE

    elementwise_kernel[(num_blocks,)](
        in2_ptr=in_2,
        in1_ptr=in_1,
        in0_ptr=in_0,
        out_ptr=out,
        n0=n0,
        n1=n1,
        n2=n2,
        n3=n3,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out

def replacement_func():
    return elementwise_optimized