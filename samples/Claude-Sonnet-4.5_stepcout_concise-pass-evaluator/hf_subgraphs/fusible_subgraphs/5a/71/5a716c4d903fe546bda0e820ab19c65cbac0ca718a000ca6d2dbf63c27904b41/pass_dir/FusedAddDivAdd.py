import torch
import triton
import triton.language as tl

def pattern(in_0, in_1, in_2, in_3):
    """Match the fused adds and division pattern"""
    tmp_0 = in_0 + in_3
    tmp_1 = tmp_0 + in_2
    tmp_2 = tmp_1 / 8.0
    tmp_3 = tmp_2 + in_1
    return tmp_3

def replacement_args(in_0, in_1, in_2, in_3):
    return (in_0, in_1, in_2, in_3)

@triton.jit
def fused_add_div_add_kernel_flat(
    in_0_ptr, in_1_ptr, in_2_ptr, in_3_ptr, out_ptr,
    n_elements,
    stride_b1, stride_h1, stride_r1, stride_c1,
    batch, heads, seq_len_row, seq_len_col,
    BLOCK_SIZE: tl.constexpr,
):
    """Fuse: (in_0 + in_3 + in_2) / 8.0 + in_1 - flattened version"""
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load from in_0, in_2, in_3 (contiguous in flattened view)
    v0 = tl.load(in_0_ptr + offsets, mask=mask, other=0.0)
    v2 = tl.load(in_2_ptr + offsets, mask=mask, other=0.0)
    v3 = tl.load(in_3_ptr + offsets, mask=mask, other=0.0)
    
    # For in_1, we need to compute the proper offset considering broadcasting
    # in_1 has shape [batch, 1, 1, seq_len_col], so we need the column index
    # from the flattened offset
    col_idx = offsets % seq_len_col
    batch_idx = offsets // (heads * seq_len_row * seq_len_col)
    
    # Compute offset in in_1
    off_1 = batch_idx * stride_b1 + col_idx * stride_c1
    v1 = tl.load(in_1_ptr + off_1, mask=mask, other=0.0)
    
    # Compute fused operation
    result = (v0 + v3 + v2) / 8.0 + v1
    
    # Store output
    tl.store(out_ptr + offsets, result, mask=mask)

@torch.fx.wrap
def fused_add_div_add(in_0, in_1, in_2, in_3):
    """Wrapper function to launch the fused kernel"""
    batch, heads, seq_len_row, seq_len_col = in_0.shape
    n_elements = in_0.numel()
    out = torch.empty_like(in_0)
    
    # Use smaller block size for better occupancy on smaller inputs
    BLOCK_SIZE = 512
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    fused_add_div_add_kernel_flat[grid](
        in_0, in_1, in_2, in_3, out,
        n_elements,
        in_1.stride(0), in_1.stride(1), in_1.stride(2), in_1.stride(3),
        batch, heads, seq_len_row, seq_len_col,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out

def replacement_func():
    return fused_add_div_add