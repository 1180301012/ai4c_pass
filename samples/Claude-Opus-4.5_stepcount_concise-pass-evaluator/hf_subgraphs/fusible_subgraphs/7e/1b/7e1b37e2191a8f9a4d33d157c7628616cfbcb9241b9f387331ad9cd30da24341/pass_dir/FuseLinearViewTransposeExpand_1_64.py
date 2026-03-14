import torch
import triton
import triton.language as tl

# Pattern matching for graph 0: just the expand part
def pattern(in_2):
    tmp_4 = in_2[slice(None, None, None), slice(None, None, None), None, slice(None, None, None), slice(None, None, None)]
    tmp_5 = tmp_4.expand(1, 4, 4, 64, 128)
    return tmp_5

def replacement_args(in_2):
    return (in_2,)

@triton.jit
def expand_kernel_1_64(
    in_ptr, out_ptr,
    n_elements,
    in_stride_0, in_stride_1, in_stride_2, in_stride_3,
    dim0, dim1, dim2, dim3, dim4,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    idx = offsets
    i4 = idx % dim4
    idx = idx // dim4
    i3 = idx % dim3
    idx = idx // dim3
    i2 = idx % dim2
    idx = idx // dim2
    i1 = idx % dim1
    i0 = idx // dim1
    
    # Input has shape [batch, n_heads, seq, head_dim], output has [batch, n_heads, repeat, seq, head_dim]
    in_idx = i0 * in_stride_0 + i1 * in_stride_1 + i3 * in_stride_2 + i4 * in_stride_3
    
    val = tl.load(in_ptr + in_idx, mask=mask, other=0.0)
    tl.store(out_ptr + offsets, val, mask=mask)

@torch.fx.wrap
def optimized_expand_1_64(in_2):
    batch = in_2.shape[0]
    num_heads = in_2.shape[1]
    seq = in_2.shape[2]
    head_dim = in_2.shape[3]
    repeat = 4
    
    tmp_5 = torch.empty((batch, num_heads, repeat, seq, head_dim), dtype=in_2.dtype, device=in_2.device)
    
    n_elements = batch * num_heads * repeat * seq * head_dim
    BLOCK_SIZE = 1024
    num_programs = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    expand_kernel_1_64[(num_programs,)](
        in_2, tmp_5,
        n_elements,
        in_2.stride(0), in_2.stride(1), in_2.stride(2), in_2.stride(3),
        batch, num_heads, repeat, seq, head_dim,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return tmp_5

def replacement_func():
    return optimized_expand_1_64