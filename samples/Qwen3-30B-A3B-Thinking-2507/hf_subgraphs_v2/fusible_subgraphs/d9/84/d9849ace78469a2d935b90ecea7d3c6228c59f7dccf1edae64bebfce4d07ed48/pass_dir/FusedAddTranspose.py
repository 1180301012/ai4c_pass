import torch
import triton
import triton.language as tl

def pattern(in_0, in_1):
    temp = in_1 + in_0
    in_2 = temp
    tmp_2 = in_2.transpose(1, 2)
    return tmp_2

def replacement_args(in_0, in_1):
    return (in_0, in_1)

@triton.jit
def fused_add_transpose_kernel(
    in_0_ptr,
    in_1_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    block_id = tl.program_id(0)
    thread_id = tl.thread_id(0)
    offset = block_id * BLOCK_SIZE + thread_id
    if offset >= n_elements:
        return
    
    in_1_offset = thread_id * 19 + block_id
    in_0_offset = thread_id
    val_in1 = tl.load(in_1_ptr + in_1_offset)
    val_in0 = tl.load(in_0_ptr + in_0_offset)
    out_val = val_in1 + val_in0
    tl.store(out_ptr + offset, out_val)

@torch.fx.wrap
def fused_add_transpose(in_0, in_1):
    N = 19 * 128
    BLOCK_SIZE = 128
    num_blocks = (N + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    out = torch.empty([1, 19, 128], device=in_0.device, dtype=in_0.dtype)
    out = out.contiguous()
    
    in_0_ptr = in_0.data_ptr()
    in_1_ptr = in_1.data_ptr()
    out_ptr = out.data_ptr()
    
    fused_add_transpose_kernel[(num_blocks,)](
        in_0_ptr=in_0_ptr,
        in_1_ptr=in_1_ptr,
        out_ptr=out_ptr,
        n_elements=N,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return out

def replacement_func():
    return fused_add_transpose