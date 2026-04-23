import torch
import triton
import triton.language as tl

def pattern(in_0, in_1):
    tmp_0 = in_0[1]
    tmp_1 = in_0[0]
    tmp_2 = in_1.index_select(0, tmp_1)
    return (tmp_0, tmp_2)

def replacement_args(in_0, in_1):
    tmp_0 = in_0[1]
    tmp_1 = in_0[0]
    return (tmp_0, in_1, tmp_1)

@triton.jit
def index_select_kernel(
    x_ptr,
    indices_ptr,
    out_ptr,
    n_indices,
    BLOCK_SIZE: tl.constexpr = 128,
):
    block_id = tl.program_id(0)
    start_idx = block_id * BLOCK_SIZE
    end_idx = min(start_idx + BLOCK_SIZE, n_indices)
    
    tid = tl.thread_id(0)
    idx_pos = start_idx + tid
    if idx_pos >= end_idx:
        return
    
    index_val = tl.load(indices_ptr + idx_pos)
    row_start = index_val * 16
    for j in range(16):
        x_val = tl.load(x_ptr + row_start + j)
        tl.store(out_ptr + idx_pos * 16 + j, x_val)

@torch.fx.wrap
def index_select_kernel_wrapper(tmp_0, in_1, tmp_1):
    n_indices = tmp_1.shape[0]
    out = torch.empty((n_indices, 16), dtype=in_1.dtype, device=in_1.device)
    
    BLOCK_SIZE = 128
    num_blocks = (n_indices + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    index_select_kernel[(num_blocks,)](
        x_ptr=in_1.data_ptr(),
        indices_ptr=tmp_1.data_ptr(),
        out_ptr=out.data_ptr(),
        n_indices=n_indices,
    )
    return (tmp_0, out)

def replacement_func():
    return index_select_kernel_wrapper